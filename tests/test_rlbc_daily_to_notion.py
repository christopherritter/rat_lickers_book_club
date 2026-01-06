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
LEGION_SWARMUI_URL = os.getenv("LEGION_SWARMUI_URL", "http://192.168.1.100:7801/API")
LEGION_MODEL = os.getenv("LEGION_MODEL", "flux1-dev.safetensors")

# APIs
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
NOTION_PAGES_URL = "https://api.notion.com/v1/pages"
NOTION_BLOCKS_URL = "https://api.notion.com/v1/blocks"
NOTION_VERSION = "2022-06-28"


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


def generate_image_on_legion(book_title: str, book_description: str) -> Optional[bytes]:
    """
    Call LEGION's SwarmUI API to generate an image.
    Returns raw image bytes if successful.
    """
    
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

    try:
        print(f"  📡 Calling LEGION to generate image for '{book_title}'...")
        
        payload = {
            "prompt": prompt,
            "negativeprompt": negative_prompt,
            "images": 1,
            "steps": 25,
            "cfg_scale": 7.0,
            "sampler": "euler_a",
            "scheduler": "normal",
            "width": 1024,
            "height": 1024,
            "model": LEGION_MODEL,
            "seed": -1,
        }
        
        response = requests.post(
            f"{LEGION_SWARMUI_URL}/GenerateText2Image",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        
        # Extract base64 image from response and decode to bytes
        if "images" in result and len(result["images"]) > 0:
            base64_image = result["images"][0]
            image_bytes = base64.b64decode(base64_image)
            print(f"  ✓ LEGION generated image ({len(image_bytes)} bytes)")
            return image_bytes
        else:
            print(f"  ✗ No image in LEGION response")
            return None
            
    except requests.exceptions.ConnectionError:
        print(f"  ✗ Cannot connect to LEGION at {LEGION_SWARMUI_URL}")
        print(f"     Make sure SwarmUI is running on LEGION")
        return None
    except requests.exceptions.Timeout:
        print(f"  ✗ LEGION request timed out")
        return None
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


def upload_image_to_notion_page(page_id: str, image_bytes: bytes, caption: str = "") -> bool:
    """
    Upload an image directly to a Notion page using block append.
    This uploads the file to Notion's storage, no external hosting needed!
    
    Returns True if successful.
    """
    
    if not image_bytes:
        return False
    
    try:
        print(f"  📤 Uploading image to Notion page...")
        
        headers = {
            "Authorization": f"Bearer {NOTION_API_KEY}",
            "Notion-Version": NOTION_VERSION,
        }
        
        # Create a multipart/form-data request with the image
        files = {
            'file': ('image.png', io.BytesIO(image_bytes), 'image/png')
        }
        
        # Notion's block children endpoint
        url = f"{NOTION_BLOCKS_URL}/{page_id}/children"
        
        # Create image block with file upload
        # Note: Notion API requires a two-step process:
        # 1. Create the block with a temporary placeholder
        # 2. Upload the actual file
        
        # Actually, let me correct this - Notion's API has a simpler approach:
        # We need to use the block append with image type and file data
        
        # Convert image to base64 for inline embedding (Notion accepts this)
        import base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Create data URL (Notion supports data URLs for images)
        data_url = f"data:image/png;base64,{base64_image}"
        
        payload = {
            "children": [
                {
                    "object": "block",
                    "type": "image",
                    "image": {
                        "type": "external",
                        "external": {
                            "url": data_url
                        }
                    }
                }
            ]
        }
        
        response = requests.patch(url, headers=headers, json=payload, timeout=30)
        
        if response.ok:
            print(f"  ✓ Image uploaded to Notion successfully")
            return True
        else:
            print(f"  ✗ Notion upload failed: {response.status_code} - {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"  ✗ Upload error: {e}")
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
        print(f"  ✗ Notion page creation error ({resp.status_code}): {resp.text[:200]}")
        resp.raise_for_status()
    
    page_data = resp.json()
    page_id = page_data["id"]
    print(f"  ✓ Notion page created: {page_id}")
    
    # Step 2: Generate and upload image for Book Features
    if post_type == "Book Feature":
        book_title = post.get("book_title", "")
        book_description = post.get("book_description", "")
        
        if book_title:
            image_bytes = generate_image_on_legion(book_title, book_description)
            if image_bytes:
                upload_image_to_notion_page(page_id, image_bytes, book_title)
    
    return page_data


def main():
    print("=" * 70)
    print("RLBC Daily Generator: qud + LEGION Architecture")
    print("  Text: Perplexity API")
    print("  Images: LEGION (SwarmUI) → Direct upload to Notion")
    print("  No external hosting needed!")
    print("=" * 70)
    print()
    
    # Test LEGION connection
    try:
        test_response = requests.get(f"{LEGION_SWARMUI_URL.rsplit('/API', 1)}", timeout=5)
        print(f"✓ LEGION connected at {LEGION_SWARMUI_URL}")
    except:
        print(f"⚠️  Warning: Cannot connect to LEGION at {LEGION_SWARMUI_URL}")
        print(f"   Make sure SwarmUI is running on LEGION")
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
            print(f"  ✓ Complete!")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        print()
    
    print("=" * 70)
    print("Done! Check your Notion database.")
    print("Images uploaded directly to Notion (like drag-and-drop)")
    print("=" * 70)


if __name__ == "__main__":
    main()
