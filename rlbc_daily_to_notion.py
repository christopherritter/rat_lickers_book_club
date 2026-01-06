import os
import json
import datetime
import base64
import io
from typing import List, Dict, Any, Optional
import requests
from dotenv import load_dotenv

load_dotenv()

# API Keys
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
RLBC_DATABASE_ID = os.getenv("RLBC_DATABASE_ID")
RLBC_TITLE_PROPERTY = os.getenv("RLBC_TITLE_PROPERTY")

# LEGION (Image Generation Server)
LEGION_SWARMUI_URL = os.getenv("LEGION_SWARMUI_URL", "http://192.168.0.227:7861")
LEGION_MODEL = os.getenv("LEGION_MODEL", "Qwen_Image_Edit_2511_Quant_Scaled_FP8")
# Default generation settings (can be overridden via env vars)
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
# LoRAs: comma-separated name:weight pairs
_default_loras = os.getenv("LEGION_LORAS", "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-fp32:0.9,Qwen_LoRA_Skin_Fix_v2:0.6")
LEGION_LORAS = []
for pair in _default_loras.split(","):
    if not pair.strip():
        continue
    name, weight = pair.rsplit(":", 1)
    try:
        w = float(weight)
    except Exception:
        w = 1.0
    LEGION_LORAS.append({"name": name.strip(), "weight": w})

# APIs
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
NOTION_PAGES_URL = "https://api.notion.com/v1/pages"
NOTION_BLOCKS_URL = "https://api.notion.com/v1/blocks"
NOTION_VERSION = "2022-06-28"

# Last generated image source (if SwarmUI returns a path we store the public URL here)
LAST_SWARMUI_IMAGE_URL: Optional[str] = None


# ============================================================================
# SwarmUI Session Management (NEW)
# ============================================================================

def get_swarmui_session() -> Optional[str]:
    """
    Get a session ID from SwarmUI.
    This is REQUIRED for all other SwarmUI API calls.
    """
    try:
        print(f" 🔐 Requesting SwarmUI session...")
        response = requests.post(
            f"{LEGION_SWARMUI_URL}/API/GetNewSession",
            json={},
            timeout=10
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
    We construct the full URL and download the actual image file. If successful, we store
    the full URL into LAST_SWARMUI_IMAGE_URL so upload can use the external URL instead of embedding bytes.
    """
    global LAST_SWARMUI_IMAGE_URL
    try:
        image_url = f"{LEGION_SWARMUI_URL}/{image_path}"
        print(f" 📥 Downloading image from: {image_url}")
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        image_bytes = response.content
        print(f" ✓ Downloaded image ({len(image_bytes)} bytes)")

        # Store the public URL for potential external upload to Notion (avoids large payloads)
        LAST_SWARMUI_IMAGE_URL = image_url
        return image_bytes
    except Exception as e:
        print(f" ✗ Failed to download image: {e}")
        return None


# ============================================================================
# Original Functions (with fixes applied)
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


def generate_image_on_legion(book_title: str, book_description: str) -> (Optional[bytes], Optional[str]):
    """
    Call LEGION's SwarmUI API to generate an image.

    Behavior:
    - Acquire a session with `GetNewSession`.
    - Try one or more generation endpoints (primary then fallbacks).
    - Handle returned image values which may be either a path (download) or a base64/data URL.

    Returns a tuple: (raw image bytes or None, public image URL if available else None).
    """
    global LAST_SWARMUI_IMAGE_URL
    # Step 1: Get a session ID (REQUIRED!)
    session_id = get_swarmui_session()
    if not session_id:
        return None, None

    # Craft the prompt for "politely dangerous" aesthetic
    prompt = f"""photorealistic book shelfie photograph, pristine white bookshelf in unnaturally perfect suburban home,
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

    # Prepare LoRAs as strings "name:weight"
    loras_as_strings = [f"{l['name']}:{l['weight']}" for l in LEGION_LORAS if l.get('name')]

    # Minimal payload: prefer SwarmUI's current settings (let server apply defaults)
    minimal_payload = {
        "session_id": session_id,
        "prompt": prompt,
        "negativeprompt": negative_prompt,
        "images": 1,
    }

    # Fallback payload (use user-provided/default settings) - only used if server requests them
    fallback_payload = {
        "session_id": session_id,
        "prompt": prompt,
        "negativeprompt": negative_prompt,
        "images": 1,
        "steps": LEGION_STEPS,
        "cfg_scale": LEGION_CFG_SCALE,
        "sampler": "euler_a",
        "scheduler": "normal",
        "width": LEGION_WIDTH,
        "height": LEGION_HEIGHT,
        "model": LEGION_MODEL,
        # Flatten refiner settings to explicit keys (SwarmUI warns on nested 'refiner')
        "refiner_control_percentage": LEGION_REFINER_CONTROL_PERCENTAGE,
        "refiner_method": LEGION_REFINER_METHOD,
        "refiner_upscale": LEGION_REFINER_UPSCALE,
        "refiner_upscale_method": LEGION_REFINER_UPSCALE_METHOD,
        "refiner_steps": LEGION_REFINER_STEPS,
        "automatic_vae": LEGION_AUTOMATIC_VAE,
        "loras": loras_as_strings,
    }

    # Start with cleaned fallback payload (includes model and defaults, but no LoRAs/refiner keys)
    fallback_clean = dict(fallback_payload)
    fallback_clean.pop("loras", None)
    for k in ["refiner_control_percentage", "refiner_method", "refiner_upscale", "refiner_upscale_method", "refiner_steps"]:
        fallback_clean.pop(k, None)

    payload = fallback_clean.copy()

    # Try primary endpoint, then fallback to alternate endpoint(s)
    endpoints_to_try = [f"{LEGION_SWARMUI_URL}/API/GenerateText2Image", f"{LEGION_SWARMUI_URL}/Text2Image"]

    for endpoint in endpoints_to_try:
        try:
            print(f" 📡 Calling LEGION endpoint: {endpoint} for '{book_title}'...")
            response = requests.post(endpoint, json=payload, timeout=120)

            # If not a successful HTTP response, check for known parameter errors and retry without them
            if not response.ok:
                text = getattr(response, "text", "") or ""
                print(f" ✗ Endpoint {endpoint} returned HTTP {response.status_code}: {text[:200]}")

                lowered = text.lower()

                # If the server complains about the model specifically, retry using the full fallback settings
                if ("invalid value for parameter model" in lowered or "invalid model value" in lowered or
                    "are you sure that model name is correct" in lowered or "no model input given" in lowered or
                    "did your ui load properly" in lowered):
                    print(" ⚠️ Server reports model is invalid or missing; retrying with user-provided defaults (fallback payload)...")
                    try:
                        retry_resp = requests.post(endpoint, json=fallback_payload, timeout=120)
                        if retry_resp.ok:
                            response = retry_resp
                        else:
                            print(f" ✗ Retry with fallback payload failed: {retry_resp.status_code}: {getattr(retry_resp, 'text', '')[:200]}")

                            # If fallback attempt failed due to LoRAs, try again without LoRAs
                            text2 = (getattr(retry_resp, 'text', '') or '').lower()
                            if "loras" in text2 or "lora" in text2:
                                print(" ⚠️ Fallback payload rejected by LoRAs; retrying fallback payload without LoRAs...")
                                cleaned = dict(fallback_payload)
                                cleaned.pop("loras", None)
                                try:
                                    retry2 = requests.post(endpoint, json=cleaned, timeout=120)
                                    if retry2.ok:
                                        response = retry2
                                    else:
                                        print(f" ✗ Retry without LoRAs also failed: {retry2.status_code}")
                                except Exception as e:
                                    print(f" ✗ Retry without LoRAs error: {e}")

                            # If fallback attempt failed due to refiner keys, try again without them
                            if "refiner" in text2 or "unrecognized" in text2:
                                print(" ⚠️ Fallback payload rejected refiner settings, retrying fallback payload without refiner keys...")
                                cleaned = dict(fallback_payload)
                                for k in ["refiner_control_percentage", "refiner_method", "refiner_upscale", "refiner_upscale_method", "refiner_steps"]:
                                    cleaned.pop(k, None)
                                try:
                                    retry2 = requests.post(endpoint, json=cleaned, timeout=120)
                                    if retry2.ok:
                                        response = retry2
                                    else:
                                        print(f" ✗ Retry without refiner also failed: {retry2.status_code}")
                                except Exception as e:
                                    print(f" ✗ Retry without refiner error: {e}")

                    except Exception as e:
                        print(f" ✗ Retry with fallback payload error: {e}")

                # Retry heuristics: remove LoRAs if server complains about LoRAs
                if "loras" in lowered or "lora" in lowered or ("invalid value for parameter loras" in lowered):
                    print(" ⚠️ Server rejected LoRAs, retrying without LoRAs...")
                    cleaned = dict(payload)
                    cleaned.pop("loras", None)
                    try:
                        retry_resp = requests.post(endpoint, json=cleaned, timeout=120)
                        if retry_resp.ok:
                            response = retry_resp
                        else:
                            print(f" ✗ Retry without LoRAs failed: {retry_resp.status_code}")
                            continue
                    except Exception as e:
                        print(f" ✗ Retry without LoRAs error: {e}")
                        continue

                # Retry heuristics: remove refiner keys if server complains about refiner
                if "refiner" in lowered or "unrecognized" in lowered:
                    print(" ⚠️ Server rejected refiner settings, retrying without refiner keys...")
                    cleaned = dict(payload)
                    for k in ["refiner_control_percentage", "refiner_method", "refiner_upscale", "refiner_upscale_method", "refiner_steps"]:
                        cleaned.pop(k, None)
                    try:
                        retry_resp = requests.post(endpoint, json=cleaned, timeout=120)
                        if retry_resp.ok:
                            response = retry_resp
                        else:
                            print(f" ✗ Retry without refiner failed: {retry_resp.status_code}")
                            continue
                    except Exception as e:
                        print(f" ✗ Retry without refiner error: {e}")
                        continue

                # If still not ok, move to next endpoint
                if not response.ok:
                    continue

            result = response.json()
            images = result.get("images", [])
            if not images:
                print(f" ✗ No images field or empty images from {endpoint}")
                continue

            first = images[0]

            # If it looks like a data URL, extract base64 and decode
            if isinstance(first, str) and first.startswith("data:"):
                try:
                    b64 = first.split(",", 1)[1]
                    image_bytes = base64.b64decode(b64)
                    print(f" ✓ Received data URL image ({len(image_bytes)} bytes)")
                    # clear any previously-stored SwarmUI URL to avoid reusing stale values
                    LAST_SWARMUI_IMAGE_URL = None
                    return image_bytes, None
                except Exception:
                    print(f" ✗ Failed to decode data URL from {endpoint}")
                    continue

            # If it's likely base64 (no slashes, long string), try decoding
            if isinstance(first, str) and "/" not in first and len(first) > 100:
                try:
                    image_bytes = base64.b64decode(first)
                    print(f" ✓ Decoded base64 image from {endpoint} ({len(image_bytes)} bytes)")
                    LAST_SWARMUI_IMAGE_URL = None
                    return image_bytes, None
                except Exception:
                    # Not valid base64 - treat as path
                    pass

            # Otherwise, treat as a path to download
            image_bytes = download_image_from_swarmui(first)
            if image_bytes:
                # download_image_from_swarmui stores the public URL in LAST_SWARMUI_IMAGE_URL
                return image_bytes, LAST_SWARMUI_IMAGE_URL
            else:
                print(f" ✗ Failed to download image path returned by {endpoint}: {first}")

        except requests.exceptions.ConnectionError:
            print(f" ✗ Cannot connect to LEGION at {endpoint}")
            continue
        except requests.exceptions.Timeout:
            print(f" ✗ LEGION request to {endpoint} timed out")
            continue
        except Exception as e:
            print(f" ✗ Error calling {endpoint}: {e}")
            continue

    # All endpoints failed
    print(" ✗ All LEGION endpoints failed to produce an image")
    return None, None


def upload_image_to_notion_page(page_id: str, image_bytes: Optional[bytes] = None, caption: str = "", external_url: Optional[str] = None) -> bool:
    """
    Upload an image to a Notion page using block append.

    Prefer to use an external URL (Notion will fetch it) to avoid sending large inline data URLs.
    If `external_url` is provided, that URL will be used. Otherwise, `image_bytes` will be encoded and embedded.

    Returns True if successful.
    """
    if not image_bytes and not external_url:
        return False

    try:
        print(f" 📤 Uploading image to Notion page...")
        headers = {
            "Authorization": f"Bearer {NOTION_API_KEY}",
            "Notion-Version": NOTION_VERSION,
        }

        # Notion's block children endpoint
        url = f"{NOTION_BLOCKS_URL}/{page_id}/children"

        if external_url:
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
            # Convert image to base64 for inline embedding (Notion accepts this)
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
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

        response = requests.patch(url, headers=headers, json=payload, timeout=30)

        if response.ok:
            print(f" ✓ Image uploaded to Notion successfully")
            return True
        else:
            print(f" ✗ Notion upload failed: {response.status_code} - {response.text[:200]}")
            return False

    except Exception as e:
        print(f" ✗ Upload error: {e}")
        return False


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
    """Convert formatted text into Notion blocks (without image - we'll add that separately)."""
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

    if content.startswith("```"):
        start = content.find("{")
        end = content.rfind("}")
        content = content[start : end + 1]

    parsed = json.loads(content)
    return parsed.get("posts", [])


def create_notion_page_from_post(post: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create Notion page with text from Perplexity and image from LEGION.
    Images are uploaded directly to Notion - no external hosting needed!
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
                # If SwarmUI provided a public URL for the image, prefer sending that to Notion to avoid large payloads
                external = image_url
                upload_image_to_notion_page(page_id, image_bytes, book_title, external_url=external)

    return page_data


def main():
    print("=" * 70)
    print("RLBC Daily Generator: Perplexity + LEGION Architecture")
    print(" Text: Perplexity API")
    print(" Images: LEGION (SwarmUI) → Direct upload to Notion")
    print(" No external hosting needed!")
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
    print("Images uploaded directly to Notion (like drag-and-drop)")
    print("=" * 70)


if __name__ == "__main__":
    main()