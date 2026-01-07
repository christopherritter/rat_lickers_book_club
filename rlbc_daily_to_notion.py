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
import re

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
LEGION_REFINER_UPSCALE_METHOD = os.getenv(
    "LEGION_REFINER_UPSCALE_METHOD", "4xRealWebPhoto_v4_dat2.safetensors"
)
LEGION_AUTOMATIC_VAE = os.getenv("LEGION_AUTOMATIC_VAE", "true").lower() in ("1", "true", "yes")
LEGION_REFINER_STEPS = int(os.getenv("LEGION_REFINER_STEPS", "7"))
LEGION_REQUEST_TIMEOUT = int(os.getenv("LEGION_REQUEST_TIMEOUT", "600"))

# Fallback candidates for refiner/upscaler methods when the configured one is rejected by the server.
# These are tried in order; the common compatible candidate 'pixel-lanczos' is first because some
# SwarmUI deployments accept it widely. If none work, the code will retry without refiner settings.
FALLBACK_REFINER_CANDIDATES = ["pixel-lanczos", "latent-bicubic", f"model-{LEGION_REFINER_UPSCALE_METHOD}"]

# LoRA Weights configuration
LEGION_LORA_WEIGHTS: Dict[str, float] = {}
_lora_weights_str = os.getenv(
    "LEGION_LORA_WEIGHTS",
    "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-fp32:0.9,Qwen_LoRA_Skin_Fix_v2:0.6",
)
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
    "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-fp32,Qwen_LoRA_Skin_Fix_v2",
)
LEGION_LORAS: List[str] = [name.strip() for name in _default_loras.split(",") if name.strip()]


def build_weighted_loras() -> List[str]:
    """
    Build LoRAs in the same format that already worked (list of strings),
    but with optional ":weight" suffix applied.
    """
    weighted: List[str] = []
    for name in LEGION_LORAS:
        w = LEGION_LORA_WEIGHTS.get(name, 1.0)
        if abs(w - 1.0) < 1e-6:
            weighted.append(name)
        else:
            weighted.append(f"{name}:{w}")
    return weighted


# API Endpoints
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
NOTION_PAGES_URL = "https://api.notion.com/v1/pages"
NOTION_BLOCKS_URL = "https://api.notion.com/v1/blocks"
NOTION_VERSION = "2022-06-28"

# Timeouts
NOTION_UPLOAD_TIMEOUT = int(os.getenv("NOTION_UPLOAD_TIMEOUT", "300"))

# Debugging: when true, print the final outgoing payloads sent to LEGION (useful for diagnosing server rejections)
LEGION_PAYLOAD_DEBUG = os.getenv("LEGION_PAYLOAD_DEBUG", "false").lower() in ("1", "true", "yes")

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
            timeout=min(LEGION_REQUEST_TIMEOUT, 30),
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
    external_url: Optional[str] = None,
) -> bool:
    """
    Upload an image to a Notion page using block append.
    Prefers external URL to avoid large inline data URLs.
    """
    if not image_bytes and not external_url:
        return False
    try:
        print(" 📤 Uploading image to Notion page...")
        headers = {
            "Authorization": f"Bearer {NOTION_API_KEY}",
            "Notion-Version": NOTION_VERSION,
        }
        url = f"{NOTION_BLOCKS_URL}/{page_id}/children"
        if external_url:
            payload = {
                "children": [
                    {
                        "object": "block",
                        "type": "image",
                        "image": {
                            "type": "external",
                            "external": {"url": external_url},
                        },
                    }
                ]
            }
        else:
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            data_url = f"data:image/png;base64,{base64_image}"
            payload = {
                "children": [
                    {
                        "object": "block",
                        "type": "image",
                        "image": {
                            "type": "external",
                            "external": {"url": data_url},
                        },
                    }
                ]
            }
        response = requests.patch(url, headers=headers, json=payload, timeout=NOTION_UPLOAD_TIMEOUT)
        if response.ok:
            print(" ✓ Image uploaded to Notion successfully")
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
    Returns: (image_bytes or None, public_url or None)
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

    # Step 3: LoRAs – start from last known-good format (list of strings), now with optional weights
    loras_names_only = [name for name in LEGION_LORAS]
    loras_weighted = build_weighted_loras()
    loras_comma = ",".join(loras_weighted)
    prompt = base_prompt

    print(f" 📋 LoRAs (names only): {loras_names_only}")
    print(f" 📋 LoRA weights: {LEGION_LORA_WEIGHTS}")
    print(f" 📋 LoRAs (weighted strings): {loras_weighted}")

    # Step 4: Normalize refiner method
    def normalize_refiner_method(m: str) -> str:
        if not m:
            return m
        s = m.lower()
        if "post" in s and ("apply" in s or "postapply" in s):
            return "PostApply"
        if "stepswap" in s:
            return "StepSwap"
        if "noisy" in s:
            return "StepSwapNoisy"
        return m

    # Step 5: Build flat payload
    # Primary: LoRAs as a sanitized comma-separated weighted string (name:weight,name2:weight)
    loras_weight_string = loras_comma.replace("\n", "").replace("|||", ",")
    loras_weight_string = ",".join([p.strip() for p in loras_weight_string.split(",") if p.strip()])

    # Build parallel weights list to match the order of LEGION_LORAS
    loras_weights_list: List[float] = []
    for name in loras_names_only:
        w = LEGION_LORA_WEIGHTS.get(name, 1.0)
        # keep integers when appropriate
        if isinstance(w, float) and w.is_integer():
            loras_weights_list.append(int(w))
        else:
            loras_weights_list.append(w)

    # Also prepare names-only comma string for servers that expect a string
    loras_names_string = ",".join(loras_names_only)
    loras_weights_string = ",".join([str(w) for w in loras_weights_list])

    payload: Dict[str, Any] = {
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
        "images": 1,
        # Refiner
        "refiner_control_percentage": LEGION_REFINER_CONTROL_PERCENTAGE,
        "refiner_method": normalize_refiner_method(LEGION_REFINER_METHOD),
        "refiner_upscale": LEGION_REFINER_UPSCALE,
        "refiner_upscale_method": LEGION_REFINER_UPSCALE_METHOD,
        "refiner_steps": LEGION_REFINER_STEPS,
        # Other
        "automatic_vae": LEGION_AUTOMATIC_VAE,
        # LoRA keys: only send what SwarmUI expects
        "LoRAs": loras_names_only,      # keep for compatibility
        "loras": loras_names_only,      # the list SwarmUI already accepts
        "weights": loras_weights_list,  # what SwarmUI is most likely expecting
        "sigma_shift": 1,
        "preferred_dtype": "default",
    }

    endpoints_to_try = [
        f"{LEGION_SWARMUI_URL}/API/GenerateText2Image",
        f"{LEGION_SWARMUI_URL}/Text2Image",
    ]

    fallback_payload = dict(payload)

    fallback_payload.pop("cfgscale", None)
    fallback_payload["cfg_scale"] = LEGION_CFG_SCALE
    fallback_payload["sampler"] = "euler_a"
    # Only keep LoRAs, loras, and weights keys
    fallback_payload["LoRAs"] = loras_names_only
    fallback_payload["loras"] = loras_names_only
    fallback_payload["weights"] = loras_weights_list
    # Remove experimental/legacy keys if present
    for k in ["loras_weights", "loras_string", "loras_weights_string"]:
        fallback_payload.pop(k, None)

    print(" 📦 Flat parameters + 'images': 1 + top-level LoRAs in 'loras'")

    response: Optional[requests.Response] = None

    def do_post(ep: str, p: dict) -> requests.Response:
        if LEGION_PAYLOAD_DEBUG:
            try:
                print(f"🔍 LEGION payload to {ep}: {json.dumps(p, default=str)[:2000]}")
            except Exception as e:
                print(f"🔍 LEGION payload (serialization error): {e}")
        return requests.post(ep, json=p, timeout=LEGION_REQUEST_TIMEOUT)

    for endpoint in endpoints_to_try:
        print(f" 📡 Calling {endpoint}")
        try:
            response = do_post(endpoint, payload)
        except requests.exceptions.ConnectionError:
            print(f" ✗ Cannot connect to LEGION at {endpoint}")
            continue
        except requests.exceptions.Timeout:
            print(f" ✗ LEGION request to {endpoint} timed out")
            continue
        except Exception as e:
            print(f" ✗ Error calling {endpoint}: {e}")
            continue

        if not response.ok:
            error_text = getattr(response, "text", "") or ""
            print(f" ✗ HTTP {response.status_code}: {error_text[:300]}")
            lowered = error_text.lower()

            if (
                "invalid value for parameter model" in lowered
                or "invalid model value" in lowered
                or "no model input given" in lowered
                or "did your ui load properly" in lowered
            ):
                print(" ⚠️ Server reports model is invalid or missing; retrying with fallback payload...")
                try:
                    retry_resp = do_post(endpoint, fallback_payload)
                    if retry_resp.ok:
                        response = retry_resp
                    else:
                        print(
                            f" ✗ Retry with fallback payload failed: "
                            f"{retry_resp.status_code}: {getattr(retry_resp,'text','')[:200]}"
                        )
                    retry_text = getattr(retry_resp, "text", "") or ""
                    lowered = lowered + " " + retry_text.lower()
                    response = retry_resp
                except Exception as e:
                    print(f" ✗ Retry with fallback payload error: {e}")

            if ("lora" in lowered or "loras" in lowered or ("invalid value for parameter loras" in lowered)):
                print(" ⚠️ LoRAs rejected; attempting alternate LoRA formats or removing LoRAs...")

                # Normalize original into parts (strip newlines and any pipe markers)
                orig = payload.get("loras") or payload.get("LoRAs") or ""
                if isinstance(orig, list):
                    parts = [p.strip() for p in orig if isinstance(p, str) and p.strip()]
                else:
                    parts = [p.strip() for p in re.split(r",|\|\|\||\||;|\s+", str(orig).replace("\n", " ")) if p.strip()]

                # Candidate variants (try in order): comma-weighted string, space-separated, single-pipe, triple-pipe, semicolon, concatenated, list
                comma_variant = ",".join(parts)
                space_variant = " ".join(parts)
                single_pipe_variant = "|".join(parts)
                triple_pipe_variant = "|||".join(parts)
                semicolon_variant = ";".join(parts)
                concat_variant = "".join(parts)
                list_variant = parts

                variants = [comma_variant, space_variant, single_pipe_variant, triple_pipe_variant, semicolon_variant, concat_variant, list_variant]

                # As a first attempt, try submitting weights under alternative key names
                weight_key_candidates = ["loras_weights", "lora_weights", "weights", "LoRAWeights"]
                # also try weights as a comma-separated string (some servers expect strings)
                loras_weights_string = ",".join([str(w) for w in loras_weights_list])
                weight_tried_and_accepted = False

                # First: try mapping name->weight under a variety of keys
                weight_map_candidates = [
                    "loras_weights_map",
                    "loras_map",
                    "lora_weights_map",
                    "lora_map",
                    "weights_map",
                    "weightsMap",
                    "LoRAWeightsMap",
                    "loras_to_weights",
                ]

                loras_weights_map = {name: LEGION_LORA_WEIGHTS.get(name, 1.0) for name in loras_names_only}

                for wk in weight_map_candidates:
                    payload_try = dict(payload)
                    payload_try["prompt"] = base_prompt
                    payload_try[wk] = loras_weights_map
                    for k in list(payload_try.keys()):
                        if k.lower() in ("loras_comma", "loras_list"):
                            payload_try.pop(k, None)
                    try:
                        print(f"🔍 Trying weight map key: {wk} -> {json.dumps(payload_try[wk])}")
                        retry_resp = do_post(endpoint, payload_try)
                        if retry_resp.ok:
                            response = retry_resp
                            weight_tried_and_accepted = True
                            print(f" ✓ Alternate weight map key accepted: {wk}")
                            break
                        else:
                            print(f" ✗ Weight map key {wk} failed -> {retry_resp.status_code}")
                    except Exception as e:
                        print(f" ✗ Weight map key {wk} error: {e}")
                if not weight_tried_and_accepted:
                    # Second: try the older weight candidates as lists/strings
                    for wk in weight_key_candidates:
                        for val in (loras_weights_list, loras_weights_string):
                            payload_try = dict(payload)
                            payload_try["prompt"] = base_prompt
                            payload_try[wk] = val
                            for k in list(payload_try.keys()):
                                if k.lower() in ("loras_comma", "loras_list"):
                                    payload_try.pop(k, None)
                            try:
                                retry_resp = do_post(endpoint, payload_try)
                                if retry_resp.ok:
                                    response = retry_resp
                                    weight_tried_and_accepted = True
                                    print(f" ✓ Alternate weight key accepted: {wk} (as {'list' if isinstance(val, list) else 'string'})")
                                    break
                                else:
                                    print(f" ✗ Weight key {wk} failed -> {retry_resp.status_code}")
                            except Exception as e:
                                print(f" ✗ Weight key {wk} error: {e}")
                        if weight_tried_and_accepted:
                            break
                if weight_tried_and_accepted:
                    accepted = True
                else:
                    # No alternative weight key worked — try string/list variants
                    accepted = False
                    for variant in variants:
                        payload_try = dict(payload)
                        payload_try["prompt"] = base_prompt

                        if isinstance(variant, list):
                            payload_try["loras"] = variant
                            payload_try["LoRAs"] = variant
                        else:
                            payload_try["loras"] = variant
                            payload_try["LoRAs"] = variant

                        # Remove other auxiliary LoRA keys to avoid server warnings
                        for k in list(payload_try.keys()):
                            if k.lower() in ("loras_comma", "loras_list"):
                                payload_try.pop(k, None)

                        try:
                            retry_resp = do_post(endpoint, payload_try)
                            if retry_resp.ok:
                                response = retry_resp
                                accepted = True
                                if isinstance(variant, list):
                                    print(" ✓ Alternate LoRAs accepted as list-of-strings")
                                else:
                                    print(f" ✓ Alternate LoRAs accepted as string variant: {variant}")
                                break
                            else:
                                print(f" ✗ LoRA variant failed: {type(variant).__name__} -> {retry_resp.status_code}")
                        except Exception as e:
                            print(f" ✗ LoRA variant error: {e}")

                    if not accepted:
                        payload_no_loras = {k: v for k, v in payload.items() if "lora" not in k.lower()}
                        payload_no_loras["prompt"] = base_prompt
                        try:
                            retry_resp = do_post(endpoint, payload_no_loras)
                            if retry_resp.ok:
                                response = retry_resp
                            else:
                                print(f" ✗ Retry without LoRAs failed: {retry_resp.status_code}")
                                continue
                        except Exception as e:
                            print(f" ✗ Retry without LoRAs error: {e}")
                            continue

            if (
                "refiner" in lowered
                or "unrecognized" in lowered
                or "invalid value for parameter refiner model" in lowered
            ):
                print(" ⚠️ Refiner rejected; trying fallbacks or retrying without refiner keys...")
                payload_no_refiner = {k: v for k, v in payload.items() if not k.startswith("refiner")}
                try:
                    retry_resp = do_post(endpoint, payload_no_refiner)
                    if retry_resp.ok:
                        response = retry_resp
                    else:
                        print(f" ✗ Retry without refiner failed: {retry_resp.status_code}")
                    tried_candidate = False
                    for candidate in FALLBACK_REFINER_CANDIDATES:
                        cleaned = dict(payload)
                        cleaned.pop("refiner_model", None)
                        cleaned["refiner_upscale_method"] = candidate
                        try:
                            retry_resp2 = do_post(endpoint, cleaned)
                            if retry_resp2.ok:
                                response = retry_resp2
                                tried_candidate = True
                                break
                            else:
                                print(
                                    f" ✗ Fallback refiner candidate '{candidate}' failed: "
                                    f"{retry_resp2.status_code}"
                                )
                        except Exception as e:
                            print(f" ✗ Error trying refiner fallback '{candidate}': {e}")
                except Exception as e:
                    print(f" ✗ Retry without refiner error: {e}")
                    tried_candidate = False
                    for candidate in FALLBACK_REFINER_CANDIDATES:
                        cleaned = dict(payload)
                        cleaned.pop("refiner_model", None)
                        cleaned["refiner_upscale_method"] = candidate
                        try:
                            retry_resp2 = requests.post(
                                endpoint, json=cleaned, timeout=LEGION_REQUEST_TIMEOUT
                            )
                            if retry_resp2.ok:
                                response = retry_resp2
                                tried_candidate = True
                                break
                            else:
                                print(
                                    f" ✗ Fallback refiner candidate '{candidate}' failed: "
                                    f"{retry_resp2.status_code}"
                                )
                        except Exception as e:
                            print(f" ✗ Error trying refiner fallback '{candidate}': {e}")

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
            elif isinstance(first, str) and "/" not in first and len(first) > 100:
                try:
                    image_bytes = base64.b64decode(first)
                    print(f" ✓ Decoded base64 image ({len(image_bytes)} bytes)")
                    LAST_SWARMUI_IMAGE_URL = None
                    return image_bytes, None
                except Exception as e:
                    print(f" ✗ Failed to decode base64: {e}")
                    return None, None
            elif isinstance(first, str) and ("View/" in first or first.startswith("/")):
                image_bytes = download_image_from_swarmui(first)
                if image_bytes:
                    return image_bytes, LAST_SWARMUI_IMAGE_URL
                return None, None
            else:
                print(f" ✗ Unknown image format: {str(first)[:100]}")
                return None, None

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
            body_parts.extend(
                [
                    "At what point does it become your constitutional duty to support airstrikes?",
                    "How many shares does kidnapping need before it feels like freedom?",
                ]
            )
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
        blocks.append(
            {
                "object": "block",
                "type": "heading_3",
                "heading_3": {"rich_text": [{"type": "text", "text": {"content": first}}]},
            }
        )
        idx = 1
    else:
        idx = 0
    while idx < len(segments):
        seg = segments[idx]
        if seg == "Discussion questions:":
            idx += 1
            while idx < len(segments) and not segments[idx].endswith(":"):
                item = segments[idx]
                blocks.append(
                    {
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {
                            "rich_text": [{"type": "text", "text": {"content": item}}]
                        },
                    }
                )
                idx += 1
            continue
        blocks.append(
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"type": "text", "text": {"content": seg}}]},
            }
        )
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
    """Create Notion page with text from Perplexity and image from LEGION."""
    if not NOTION_API_KEY or not RLBC_DATABASE_ID:
        raise RuntimeError("NOTION_API_KEY or RLBC_DATABASE_ID is not set")
    name = post.get("name") or "RLBC Post"
    date_str = post.get("date") or datetime.date.today().isoformat()
    post_type = post.get("post_type") or "Other"
    platform = post.get("platform") or "Facebook"
    source_headlines = post.get("source_headlines") or ""
    post_text = format_post_text(
        {
            "name": name,
            "post_type": post_type,
            "post_text": post.get("post_text"),
            "source_headlines": source_headlines,
        }
    )
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_VERSION,
    }
    title_prop_name = RLBC_TITLE_PROPERTY or get_title_property_name()
    payload = {
        "parent": {"type": "database_id", "database_id": RLBC_DATABASE_ID},
        "properties": {
            title_prop_name: {"title": [{"text": {"content": name}}]},
            "Date": {"date": {"start": date_str}},
            "Post Type": {"select": {"name": post_type}},
            "Platform": {"select": {"name": platform}},
            "Status": {"select": {"name": "Draft"}},
            "Source Headlines": {"rich_text": [{"text": {"content": source_headlines}}]},
            "Post Text": {"rich_text": [{"text": {"content": post_text or ""}}]},
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
    print(" FINAL FIX: Flat params + 'images' param + LoRAs with optional weights")
    print("=" * 70)
    print()
    try:
        test_response = requests.get(f"{LEGION_SWARMUI_URL}", timeout=5)
        print(f"✓ LEGION web interface reachable at {LEGION_SWARMUI_URL}")
    except Exception as e:
        print(f"⚠️ Warning: Cannot connect to LEGION web interface: {e}")
        print(" (This is OK if SwarmUI API is still running)")
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
            print(" ✓ Complete!")
        except Exception as e:
            print(f" ✗ Error: {e}")
        print()
    print("=" * 70)
    print("Done! Check your Notion database.")
    print("=" * 70)


if __name__ == "__main__":
    main()
