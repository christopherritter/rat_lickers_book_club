"""
RLBC Daily Generator: Perplexity + LEGION + Notion Integration
FINAL FIX: All required parameters at top level + "images" parameter

This script:
1. Generates satirical book club posts using Perplexity API
2. Creates Notion database entries with formatted content
3. Generates custom images using SwarmUI API (LEGION server)
4. Uploads images directly to Notion

Requires environment variables:
  - PERPLEXITY_API_KEY
  - NOTION_API_KEY
  - RLBC_DATABASE_ID
  - LEGION_SWARMUI_URL (optional, defaults to http://192.168.0.227:7861)
"""

import os
import json
import datetime
import base64
from typing import List, Dict, Any, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

# API Keys
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
RLBC_DATABASE_ID = os.getenv("RLBC_DATABASE_ID")
RLBC_TITLE_PROPERTY = os.getenv("RLBC_TITLE_PROPERTY")

# LEGION (SwarmUI Image Generation Server)
LEGION_SWARMUI_URL = os.getenv("LEGION_SWARMUI_URL", "http://192.168.0.227:7861")
LEGION_MODEL = os.getenv("LEGION_MODEL", "Qwen_Image_Edit_2511_Quant_Scaled_FP8")

# Generation Settings
LEGION_WIDTH = int(os.getenv("LEGION_WIDTH", "1168"))
LEGION_HEIGHT = int(os.getenv("LEGION_HEIGHT", "1488"))
LEGION_STEPS = int(os.getenv("LEGION_STEPS", "4"))
LEGION_CFG_SCALE = float(os.getenv("LEGION_CFG_SCALE", "1"))
LEGION_REFINER_CONTROL_PERCENTAGE = float(os.getenv("LEGION_REFINER_CONTROL_PERCENTAGE", "0.6"))
LEGION_REFINER_METHOD = os.getenv("LEGION_REFINER_METHOD", "Post-Apply (Normal)")
LEGION_REFINER_UPSCALE = int(os.getenv("LEGION_REFINER_UPSCALE", "2"))
LEGION_REFINER_UPSCALE_METHOD = os.getenv("LEGION_REFINER_UPSCALE_METHOD", "4xRealWebPhoto_v4_dat2.safetensors")
LEGION_AUTOMATIC_VAE = os.getenv("LEGION_AUTOMATIC_VAE", "true").lower() in ("1", "true", "yes")
LEGION_REFINER_STEPS = int(os.getenv("LEGION_REFINER_STEPS", "7"))
LEGION_REQUEST_TIMEOUT = int(os.getenv("LEGION_REQUEST_TIMEOUT", "600"))

# Fallback candidates for refiner/upscaler methods when the configured one is rejected by the server.
# These are tried in order; the common compatible candidate 'pixel-lanczos' is first because some
# SwarmUI deployments accept it widely. If none work, the code will retry without refiner settings.
FALLBACK_REFINER_CANDIDATES = ["pixel-lanczos", "latent-bicubic", f"model-{LEGION_REFINER_UPSCALE_METHOD}"]

# LoRA Weights configuration
LEGION_LORA_WEIGHTS = {}
_lora_weights_str = os.getenv("LEGION_LORA_WEIGHTS", "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-fp32:0.9,Qwen_LoRA_Skin_Fix_v2:0.6")
for pair in _lora_weights_str.split(","):
    if not pair.strip():
        continue
    try:
        name, weight = pair.rsplit(":", 1)
        LEGION_LORA_WEIGHTS[name.strip()] = float(weight)
    except Exception:
        pass

# LoRAs: just the names, comma-separated
_default_loras = os.getenv(
    "LEGION_LORAS",
    "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-fp32,Qwen_LoRA_Skin_Fix_v2"
)
LEGION_LORAS = [name.strip() for name in _default_loras.split(",") if name.strip()]

# API Endpoints
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
NOTION_PAGES_URL = "https://api.notion.com/v1/pages"
NOTION_BLOCKS_URL = "https://api.notion.com/v1/blocks"
NOTION_VERSION = "2022-06-28"

# Timeouts
NOTION_UPLOAD_TIMEOUT = int(os.getenv("NOTION_UPLOAD_TIMEOUT", "300"))

# Global state
LAST_SWARMUI_IMAGE_URL: Optional[str] = None


# ============================================================================
# SwarmUI Session Management
# ============================================================================

def get_swarmui_session() -> Optional[str]:
    """
    Get a session ID from SwarmUI.
    This is REQUIRED for all other SwarmUI API calls.
    """
    try:
        print(" 🔐 Requesting SwarmUI session...")
        response = requests.post(
            f"{LEGION_SWARMUI_URL}/API/GetNewSession",
            json={},
            timeout=min(LEGION_REQUEST_TIMEOUT, 30)
        )
        response.raise_for_status()
        result = response.json()
        session_id = result.get("session_id")
        
        if session_id:
            print(f" ✓ Got session: {session_id[:16]}...")
            return session_id
        else:
            print(f" ✗ No session_id in response: {result}")
            return None
            
    except requests.exceptions.ConnectionError:
        print(f" ✗ Cannot connect to SwarmUI at {LEGION_SWARMUI_URL}")
        return None
    except Exception as e:
        print(f" ✗ Failed to get SwarmUI session: {e}")
        return None


def download_image_from_swarmui(image_path: str) -> Optional[bytes]:
    """
    Download image from SwarmUI using the returned file path.
    SwarmUI returns paths like: "View/local/raw/2024-05-19/image_name.png"
    """
    global LAST_SWARMUI_IMAGE_URL
    
    try:
        image_url = f"{LEGION_SWARMUI_URL}/{image_path}"
        print(f" 📥 Downloading image from: {image_url}")
        response = requests.get(image_url, timeout=min(LEGION_REQUEST_TIMEOUT, 300))
        response.raise_for_status()
        image_bytes = response.content
        print(f" ✓ Downloaded image ({len(image_bytes)} bytes)")
        
        # Store the public URL for Notion upload
        LAST_SWARMUI_IMAGE_URL = image_url
        return image_bytes
        
    except Exception as e:
        print(f" ✗ Failed to download image: {e}")
        return None


# ============================================================================
# Notion Helper Functions
# ============================================================================

def get_title_property_name() -> str:
    """Get title property from Notion database schema."""
    if not NOTION_API_KEY or not RLBC_DATABASE_ID:
        raise RuntimeError("NOTION_API_KEY or RLBC_DATABASE_ID is not set")
    
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": NOTION_VERSION,
    }
    
    url = f"https://api.notion.com/v1/databases/{RLBC_DATABASE_ID}"
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    
    data = resp.json()
    properties = data.get("properties", {})
    
    for prop_name, prop_def in properties.items():
        if prop_def.get("type") == "title":
            return prop_name
    
    raise RuntimeError("No title property found in Notion database.")


def upload_image_to_notion_page(
    page_id: str,
    image_bytes: Optional[bytes] = None,
    caption: str = "",
    external_url: Optional[str] = None
) -> bool:
    """
    Upload an image to a Notion page using block append.
    Prefers external URL to avoid large inline data URLs.
    
    Args:
        page_id: Notion page ID
        image_bytes: Raw image bytes (optional)
        caption: Image caption (unused)
        external_url: External URL for the image (preferred)
    
    Returns:
        True if successful, False otherwise
    """
    if not image_bytes and not external_url:
        return False
    
    try:
        print(f" 📤 Uploading image to Notion page...")
        headers = {
            "Authorization": f"Bearer {NOTION_API_KEY}",
            "Notion-Version": NOTION_VERSION,
        }
        
        url = f"{NOTION_BLOCKS_URL}/{page_id}/children"
        
        if external_url:
            # Use external URL (SwarmUI will host it)
            payload = {
                "children": [
                    {
                        "object": "block",
                        "type": "image",
                        "image": {
                            "type": "external",
                            "external": {"url": external_url}
                        }
                    }
                ]
            }
        else:
            # Embed as data URL
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            data_url = f"data:image/png;base64,{base64_image}"
            payload = {
                "children": [
                    {
                        "object": "block",
                        "type": "image",
                        "image": {
                            "type": "external",
                            "external": {"url": data_url}
                        }
                    }
                ]
            }
        
        response = requests.patch(url, headers=headers, json=payload, timeout=NOTION_UPLOAD_TIMEOUT)
        
        if response.ok:
            print(f" ✓ Image uploaded to Notion successfully")
            return True
        else:
            print(f" ✗ Notion upload failed: {response.status_code} - {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f" ✗ Upload error: {e}")
        return False


# ============================================================================
# Image Generation (SwarmUI / LEGION)
# ============================================================================

def generate_image_on_legion(book_title: str, book_description: str) -> tuple[Optional[bytes], Optional[str]]:
    """
    Call LEGION's SwarmUI API to generate an image.
    FINAL FIX: All parameters at top level + required "images" parameter!
    
    Returns:
        Tuple of (image_bytes or None, public_url or None)
    """
    global LAST_SWARMUI_IMAGE_URL
    
    # Step 1: Get session ID
    session_id = get_swarmui_session()
    if not session_id:
        return None, None
    
    # Step 2: Build base prompt
    base_prompt = f"""photorealistic book shelfie photograph, pristine white bookshelf in unnaturally perfect suburban home,
soft clinical lighting reminiscent of "Get Out" movie aesthetic, bright but cold and sterile,
book prominently displayed showing title: "{book_title}",
surrounding books with vague ominous self-help titles on spines,
subtle unsettling elements: too-perfect white flowers, wine glass with dark red wine, pearl necklace draped casually,
color palette: whites creams pastels with occasional blood-red accents,
Stepford Wives meets modern book club, performative normalcy masking authoritarian impulses,
shallow depth of field, book in sharp focus, background soft blur,
slight tilted angle as if photographed casually by book club member,
photorealistic, 8k quality, professional photography, highly detailed"""

    negative_prompt = """blurry, low quality, cartoonish, anime, illustration, drawing,
messy, cluttered, dirty, damaged books, poor lighting, overexposed, underexposed,
text errors, misspelled words, distorted text, warped perspective, multiple books in focus"""
    
    # Step 3: Prepare LoRAs as a top-level parameter (avoid embedding tags in prompt)
    # SwarmUI expectations vary across deployments. The most widely accepted format we've
    # standardized on is a pipe-delimited string of "name:weight" entries, e.g.
    # "Qwen-Image-Edit-2509-...:0.9|||Qwen_LoRA_Skin_Fix_v2:0.6".
    # Reasons:
    # - Some SwarmUI instances accept a pipe-delimited string (most common in current versions).
    # - Others accept a comma-delimited string or a list of "name:weight" strings.
    # - A few older variants accepted lists of objects (e.g., [{"name":"...","weight":0.9}]) —
    #   these typically cause parsing issues and are used only as a last resort.
    #
    # Implementation policy:
    # - Send pipe-delimited string by default (highest compatibility with modern SwarmUI).
    # - If SwarmUI rejects LoRAs, try converting to a list-of-strings form and resubmit.
    # - If still rejected, remove LoRAs and proceed without them.

    # Build canonical representations
    loras_list_of_strings = [f"{name}:{LEGION_LORA_WEIGHTS.get(name, 1.0)}" for name in LEGION_LORAS]
    loras_pipe = "|||".join(loras_list_of_strings)
    loras_comma = ",".join(loras_list_of_strings)

    prompt = base_prompt

    print(f" 📋 LoRAs (names): {LEGION_LORAS}")
    print(f" 📋 LoRA weights: {LEGION_LORA_WEIGHTS}")
    print(f" 📋 Primary LoRAs payload (pipe-delimited): {loras_pipe}")

    # Step 4: Normalize refiner method
    def normalize_refiner_method(m: str) -> str:
        if not m:
            return m
        s = m.lower()
        if 'post' in s and ('apply' in s or 'postapply' in s):
            return 'PostApply'
        if 'stepswap' in s:
            return 'StepSwap'
        if 'noisy' in s:
            return 'StepSwapNoisy'
        return m

    # Step 5: Build FLAT payload - NO rawInput wrapper!
    # ALL parameters at top level
    # CRITICAL: "images" parameter required!
    # Create both list and canonical string forms for LoRAs to maximize compatibility.
    # Some SwarmUI deployments expect a list, others a comma-delimited string, and some may use
    # the capitalized 'LoRAs' key name. Provide all common variants to avoid rejection.
    # keep an explicit comma-based string variant available
    loras_str = loras_comma

    payload = {
        "session_id": session_id,
        "prompt": prompt,
        "negativeprompt": negative_prompt,
        "model": LEGION_MODEL,
        "width": LEGION_WIDTH,
        "height": LEGION_HEIGHT,
        "steps": LEGION_STEPS,
        "cfgscale": LEGION_CFG_SCALE,
        "sampler": "euler_ancestral",
        "scheduler": "normal",
        "images": 1,  # ← CRITICAL: Number of images to generate!

        # Refiner settings
        "refiner_control_percentage": LEGION_REFINER_CONTROL_PERCENTAGE,
        "refiner_method": normalize_refiner_method(LEGION_REFINER_METHOD),
        "refiner_upscale": LEGION_REFINER_UPSCALE,
        "refiner_upscale_method": LEGION_REFINER_UPSCALE_METHOD,
        "refiner_steps": LEGION_REFINER_STEPS,

        # Other settings
        "automatic_vae": LEGION_AUTOMATIC_VAE,  # canonical key
        "autovae": LEGION_AUTOMATIC_VAE,        # legacy key (kept for compatibility)
        # Primary LoRAs format: pipe-delimited string (most SwarmUI deployments accept this)
        "loras": loras_pipe,
        # Also include common alternate names in case a deployment expects them
        "LoRAs": loras_pipe,
        "loras_comma": loras_comma,
        "loras_list": loras_list_of_strings,
        "sigma_shift": 1,
        "preferred_dtype": "default"
    }

    # Try multiple endpoints (some servers expose both API/GenerateText2Image and /Text2Image)
    endpoints_to_try = [f"{LEGION_SWARMUI_URL}/API/GenerateText2Image", f"{LEGION_SWARMUI_URL}/Text2Image"]

    # Build a fallback payload variant for servers that expect snake_case or list LoRAs
    fallback_payload = dict(payload)
    fallback_payload.pop("cfgscale", None)
    fallback_payload["cfg_scale"] = LEGION_CFG_SCALE
    fallback_payload["sampler"] = "euler_a"
    fallback_payload["loras"] = loras_list_of_strings

    print(f" 📦 Flat parameters + 'images': 1 + top-level LoRAs in 'loras'")

    response = None
    for endpoint in endpoints_to_try:
        print(f" 📡 Calling {endpoint}")
        try:
            response = requests.post(endpoint, json=payload, timeout=LEGION_REQUEST_TIMEOUT)
        except requests.exceptions.ConnectionError:
            print(f" ✗ Cannot connect to LEGION at {endpoint}")
            continue
        except requests.exceptions.Timeout:
            print(f" ✗ LEGION request to {endpoint} timed out")
            continue
        except Exception as e:
            print(f" ✗ Error calling {endpoint}: {e}")
            continue

        # Handle non-OK responses with targeted retries
        if not response.ok:
            error_text = getattr(response, 'text', '') or ''
            print(f" ✗ HTTP {response.status_code}: {error_text[:300]}")
            lowered = error_text.lower()

            # If server complains about model, try fallback payload (different keys)
            if ("invalid value for parameter model" in lowered or "invalid model value" in lowered or
                "no model input given" in lowered or "did your ui load properly" in lowered):
                print(" ⚠️ Server reports model is invalid or missing; retrying with fallback payload...")
                try:
                    retry_resp = requests.post(endpoint, json=fallback_payload, timeout=LEGION_REQUEST_TIMEOUT)
                    if retry_resp.ok:
                        response = retry_resp
                    else:
                        print(f" ✗ Retry with fallback payload failed: {retry_resp.status_code}: {getattr(retry_resp,'text','')[:200]}")
                        # Make subsequent error handling aware of the retry response text (e.g., LoRA errors)
                        retry_text = getattr(retry_resp, 'text', '') or ''
                        lowered = lowered + " " + retry_text.lower()
                        # Keep reference to the retry response so further checks operate on it
                        response = retry_resp
                except Exception as e:
                    print(f" ✗ Retry with fallback payload error: {e}")

            # If LoRAs fail, attempt alternate formats (list-of-strings) then remove
            if ("lora" in lowered or "loras" in lowered or ("invalid value for parameter loras" in lowered)):
                print(" ⚠️  LoRAs rejected; attempting alternate LoRA formats or removing LoRAs...")
                if isinstance(payload.get("loras"), str) and "|||" in payload.get("loras"):
                    payload_try = dict(payload)
                    parts = [p for p in payload.get("loras").split("|||") if p.strip()]
                    payload_try["loras"] = parts
                    payload_try["prompt"] = base_prompt
                    try:
                        retry_resp = requests.post(endpoint, json=payload_try, timeout=LEGION_REQUEST_TIMEOUT)
                        if retry_resp.ok:
                            response = retry_resp
                            print(" ✓ Alternate LoRAs (list-of-strings) accepted")
                        else:
                            print(f" ✗ Alternate LoRAs failed: {retry_resp.status_code}")
                    except Exception as e:
                        print(f" ✗ Alternate LoRAs error: {e}")

                # Final fallback: remove LoRAs entirely
                if not response.ok:
                    payload_no_loras = {k: v for k, v in payload.items() if "lora" not in k.lower()}
                    payload_no_loras["prompt"] = base_prompt
                    try:
                        retry_resp = requests.post(endpoint, json=payload_no_loras, timeout=LEGION_REQUEST_TIMEOUT)
                        if retry_resp.ok:
                            response = retry_resp
                        else:
                            print(f" ✗ Retry without LoRAs failed: {retry_resp.status_code}")
                            continue
                    except Exception as e:
                        print(f" ✗ Retry without LoRAs error: {e}")
                        continue

            # If refiner fails, prefer retrying without refiner keys first (minimal payload),
            # then attempt candidate upscalers if needed
            if ("refiner" in lowered or "unrecognized" in lowered or "invalid value for parameter refiner model" in lowered):
                print(" ⚠️  Refiner rejected; trying fallbacks or retrying without refiner keys...")

                # First, try a minimal payload with refiner keys removed
                payload_no_refiner = {k: v for k, v in payload.items() if not k.startswith("refiner")}
                try:
                    retry_resp = requests.post(endpoint, json=payload_no_refiner, timeout=LEGION_REQUEST_TIMEOUT)
                    if retry_resp.ok:
                        response = retry_resp
                    else:
                        print(f" ✗ Retry without refiner failed: {retry_resp.status_code}")
                        # If removing refiner didn't work, try candidate upscalers
                        tried_candidate = False
                        for candidate in FALLBACK_REFINER_CANDIDATES:
                            cleaned = dict(payload)
                            cleaned.pop("refiner_model", None)
                            cleaned["refiner_upscale_method"] = candidate
                            try:
                                retry_resp2 = requests.post(endpoint, json=cleaned, timeout=LEGION_REQUEST_TIMEOUT)
                                if retry_resp2.ok:
                                    response = retry_resp2
                                    tried_candidate = True
                                    break
                                else:
                                    print(f" ✗ Fallback refiner candidate '{candidate}' failed: {retry_resp2.status_code}")
                            except Exception as e:
                                print(f" ✗ Error trying refiner fallback '{candidate}': {e}")
                except Exception as e:
                    print(f" ✗ Retry without refiner error: {e}")
                    # If a network/other error occurred, still try candidate upscalers
                    tried_candidate = False
                    for candidate in FALLBACK_REFINER_CANDIDATES:
                        cleaned = dict(payload)
                        cleaned.pop("refiner_model", None)
                        cleaned["refiner_upscale_method"] = candidate
                        try:
                            retry_resp2 = requests.post(endpoint, json=cleaned, timeout=LEGION_REQUEST_TIMEOUT)
                            if retry_resp2.ok:
                                response = retry_resp2
                                tried_candidate = True
                                break
                            else:
                                print(f" ✗ Fallback refiner candidate '{candidate}' failed: {retry_resp2.status_code}")
                        except Exception as e:
                            print(f" ✗ Error trying refiner fallback '{candidate}': {e}")
        
        # If we have a successful response, process it now
        if response and response.ok:
            try:
                result = response.json()
            except Exception:
                print(f" ✗ Failed to parse JSON response from {endpoint}")
                continue

            images = result.get("images", [])

            if not images:
                print(f" ✗ No images field or empty images from {endpoint}")
                continue

            first = images[0]
            print(f" 📊 Image response: {type(first).__name__}")
        
            # Step 7: Handle different response formats
            # Format 1: Data URL
            if isinstance(first, str) and first.startswith("data:"):
                try:
                    b64 = first.split(",", 1)[1]
                    image_bytes = base64.b64decode(b64)
                    print(f" ✓ Decoded data URL image ({len(image_bytes)} bytes)")
                    LAST_SWARMUI_IMAGE_URL = None
                    return image_bytes, None
                except Exception as e:
                    print(f" ✗ Failed to decode data URL: {e}")
                    return None, None

            # Format 2: Raw base64
            elif isinstance(first, str) and "/" not in first and len(first) > 100:
                try:
                    image_bytes = base64.b64decode(first)
                    print(f" ✓ Decoded base64 image ({len(image_bytes)} bytes)")
                    LAST_SWARMUI_IMAGE_URL = None
                    return image_bytes, None
                except Exception as e:
                    print(f" ✗ Failed to decode base64: {e}")
                    return None, None

            # Format 3: File path
            elif isinstance(first, str) and ("View/" in first or first.startswith("/")):
                image_bytes = download_image_from_swarmui(first)
                if image_bytes:
                    return image_bytes, LAST_SWARMUI_IMAGE_URL
                return None, None

            else:
                print(f" ✗ Unknown image format: {str(first)[:100]}")
                return None, None



# ============================================================================
# Content Formatting
# ============================================================================

def format_post_text(post: Dict[str, Any]) -> str:
    """Format post text based on post type."""
    post_type = (post.get("post_type") or "").strip()
    name = post.get("name") or ""
    raw_text = (post.get("post_text") or "").strip()
    sources = post.get("source_headlines") or ""
    
    def sources_to_list(s: str):
        return [p.strip() for p in s.split(";") if p.strip()]
    
    header_markers = (
        "Tonight's Meeting",
        "This Week's Selection",
        "Reading Guide",
        "Banter",
    )
    
    if raw_text and any(raw_text.startswith(m) for m in header_markers):
        return raw_text
    
    if post_type == "Meeting":
        if raw_text:
            header = f"Tonight's Meeting: {name}\n\n" if name else "Tonight's Meeting\n\n"
            return header + raw_text
        
        bullets = sources_to_list(sources)
        body_parts = [
            "Ladies, please bring your thumbed‑through copies of Operation Neighborhood Regime Change.",
            "Discussion questions:",
        ]
        
        if bullets:
            body_parts.extend(bullets)
        else:
            body_parts.extend([
                "At what point does it become your constitutional duty to support airstrikes?",
                "How many shares does kidnapping need before it feels like freedom?",
            ])
        
        body_parts.append("Snacks: Anything oil‑based.")
        return "Tonight's Meeting\n\n" + "\n\n".join(body_parts)
    
    if post_type == "Book Feature":
        if raw_text:
            if raw_text.startswith("This Week's Selection"):
                return raw_text
            return f"This Week's Selection:\n\n{raw_text}"
        
        body = (
            "Back-cover blurb:\n"
            "Once you've mastered vision boards, you're ready for assembling narratives so tight "
            "your aunt genuinely believes bombing a capital is normal."
        )
        return "This Week's Selection:\n\n" + body
    
    if post_type == "Banter":
        if raw_text:
            return raw_text
        return "Just finished Eat, Pray, Invade."
    
    if "reading" in post_type.lower() or "guide" in post_type.lower():
        if raw_text:
            return raw_text
        body = "Chapter 3 asks: what color palette expresses your support for airstrikes?"
        return f"Reading Guide for {name}\n\n{body}" if name else f"Reading Guide\n\n{body}"
    
    return raw_text or name or ""


def formatted_text_to_blocks(formatted: str) -> List[Dict[str, Any]]:
    """Convert formatted text into Notion blocks."""
    if not formatted:
        return []
    
    blocks: List[Dict[str, Any]] = []
    segments = [seg.strip() for seg in formatted.split("\n\n") if seg.strip()]
    
    if not segments:
        return blocks
    
    first = segments[0]
    header_labels = ("Tonight's Meeting:", "This Week's Selection:", "Reading Guide for", "Banter:")
    
    if any(first.startswith(lbl) for lbl in header_labels):
        blocks.append({
            "object": "block",
            "type": "heading_3",
            "heading_3": {"rich_text": [{"type": "text", "text": {"content": first}}]},
        })
        idx = 1
    else:
        idx = 0
    
    # Process remaining segments
    while idx < len(segments):
        seg = segments[idx]
        
        if seg == "Discussion questions:":
            idx += 1
            while idx < len(segments) and not segments[idx].endswith(":"):
                item = segments[idx]
                blocks.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": item}}]},
                })
                idx += 1
            continue
        
        blocks.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {"rich_text": [{"type": "text", "text": {"content": seg}}]},
        })
        idx += 1
    
    return blocks


# ============================================================================
# Perplexity API
# ============================================================================

def call_perplexity_for_posts() -> List[Dict[str, Any]]:
    """Call Perplexity API to generate RLBC posts."""
    if not PERPLEXITY_API_KEY:
        raise RuntimeError("PERPLEXITY_API_KEY is not set")
    
    user_prompt = """
Generate 3-5 posts for "Rat Licker's Book Club" - a satirical project about suburban book club culture meets propaganda.

Output ONLY this JSON structure:

{
  "posts": [
    {
      "name": "short label",
      "date": "YYYY-MM-DD",
      "post_type": "Meeting | Book Feature | Banter",
      "platform": "Facebook",
      "post_text": "2-6 sentences in dry, surreal satire style",
      "source_headlines": "news headlines used",
      "book_title": "SHORT invented title (2-5 words, under 40 chars) for Book Features only",
      "book_description": "1-2 sentence satirical description for Book Features only"
    }
  ]
}

Style: Twisted self-help titles like "Eat, Pray, Invade" or "Manifesting Regime Change"
Keep book titles SHORT for image generation (2-5 words max)
"""
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }
    
    body = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": "You are a JSON-only content generator."},
            {"role": "user", "content": user_prompt.strip()},
        ],
        "temperature": 0.8,
        "max_tokens": 2000,
    }
    
    resp = requests.post(PERPLEXITY_API_URL, headers=headers, json=body, timeout=60)
    resp.raise_for_status()
    
    data = resp.json()
    content = data["choices"][0]["message"]["content"].strip()
    
    # Extract JSON from markdown code blocks if present
    if content.startswith("```"):
        start = content.find("{")
        end = content.rfind("}")
        content = content[start : end + 1]
    
    parsed = json.loads(content)
    return parsed.get("posts", [])


# ============================================================================
# Notion Page Creation
# ============================================================================

def create_notion_page_from_post(post: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create Notion page with text from Perplexity and image from LEGION.
    """
    if not NOTION_API_KEY or not RLBC_DATABASE_ID:
        raise RuntimeError("NOTION_API_KEY or RLBC_DATABASE_ID is not set")
    
    name = post.get("name") or "RLBC Post"
    date_str = post.get("date") or datetime.date.today().isoformat()
    post_type = post.get("post_type") or "Other"
    platform = post.get("platform") or "Facebook"
    source_headlines = post.get("source_headlines") or ""
    
    post_text = format_post_text({
        "name": name,
        "post_type": post_type,
        "post_text": post.get("post_text"),
        "source_headlines": source_headlines,
    })
    
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_VERSION,
    }
    
    title_prop_name = RLBC_TITLE_PROPERTY or get_title_property_name()
    
    # Step 1: Create the page with text content
    payload = {
        "parent": {
            "type": "database_id",
            "database_id": RLBC_DATABASE_ID,
        },
        "properties": {
            title_prop_name: {
                "title": [{"text": {"content": name}}]
            },
            "Date": {
                "date": {"start": date_str}
            },
            "Post Type": {
                "select": {"name": post_type}
            },
            "Platform": {
                "select": {"name": platform}
            },
            "Status": {
                "select": {"name": "Draft"}
            },
            "Source Headlines": {
                "rich_text": [{"text": {"content": source_headlines}}]
            },
            "Post Text": {
                "rich_text": [{"text": {"content": post_text or ""}}]
            },
        },
        "children": formatted_text_to_blocks(post_text),
    }
    
    resp = requests.post(NOTION_PAGES_URL, headers=headers, json=payload, timeout=30)
    
    if not resp.ok:
        print(f" ✗ Notion page creation error ({resp.status_code}): {resp.text[:200]}")
        resp.raise_for_status()
    
    page_data = resp.json()
    page_id = page_data["id"]
    print(f" ✓ Notion page created: {page_id}")
    
    # Step 2: Generate and upload image for Book Features
    if post_type == "Book Feature":
        book_title = post.get("book_title", "")
        book_description = post.get("book_description", "")
        
        if book_title:
            image_bytes, image_url = generate_image_on_legion(book_title, book_description)
            if image_bytes:
                external = image_url
                upload_image_to_notion_page(page_id, image_bytes, book_title, external_url=external)
    
    return page_data


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("RLBC Daily Generator: Perplexity + LEGION + Notion")
    print(" Text: Perplexity API (sonar-pro)")
    print(" Images: LEGION (SwarmUI) → Notion")
    print(" FINAL FIX: Flat params + 'images' param + LoRAs in prompt")
    print("=" * 70)
    print()
    
    # Test LEGION connection
    try:
        test_response = requests.get(f"{LEGION_SWARMUI_URL}", timeout=5)
        print(f"✓ LEGION web interface reachable at {LEGION_SWARMUI_URL}")
    except Exception as e:
        print(f"⚠️  Warning: Cannot connect to LEGION web interface: {e}")
        print(f" (This is OK if SwarmUI API is still running)")
    
    print()
    print("Fetching posts from Perplexity...")
    posts = call_perplexity_for_posts()
    print(f"✓ Fetched {len(posts)} posts from Perplexity")
    print()
    
    for idx, post in enumerate(posts, start=1):
        post_type = post.get("post_type", "Unknown")
        name = post.get("name", "Unnamed")
        print(f"[{idx}/{len(posts)}] {name} ({post_type})")
        
        try:
            create_notion_page_from_post(post)
            print(f" ✓ Complete!")
        except Exception as e:
            print(f" ✗ Error: {e}")
    
    print()
    print("=" * 70)
    print("Done! Check your Notion database.")
    print("=" * 70)


if __name__ == "__main__":
    main()