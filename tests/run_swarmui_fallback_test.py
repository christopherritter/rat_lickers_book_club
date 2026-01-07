import os
import sys
import json
import base64
import requests

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import rlbc_daily_to_notion as rlbc

URL = rlbc.LEGION_SWARMUI_URL
ENDPOINTS = [f"{URL}/API/GenerateText2Image", f"{URL}/Text2Image"]

print("Starting live SwarmUI fallback-first test...")

# Step 1: get session
try:
    r = requests.post(f"{URL}/API/GetNewSession", json={}, timeout=10)
    r.raise_for_status()
    sid = r.json().get("session_id")
    print("Got session:", sid)
except Exception as e:
    print("Failed to get session:", e)
    sid = None

if not sid:
    print("Cannot proceed without session")
    sys.exit(1)

prompt = "photorealistic book shelfie photograph, book prominently displayed showing title: 'Live Fallback Test'"
negative = "blurry, low quality"

loras_as_strings = [f"{name}:{rlbc.LEGION_LORA_WEIGHTS.get(name, 1.0)}" for name in rlbc.LEGION_LORAS]

fallback_full = {
    "session_id": sid,
    "prompt": prompt,
    "negativeprompt": negative,
    "images": 1,
    "steps": rlbc.LEGION_STEPS,
    "cfg_scale": rlbc.LEGION_CFG_SCALE,
    "sampler": "euler_a",
    "scheduler": "normal",
    "width": rlbc.LEGION_WIDTH,
    "height": rlbc.LEGION_HEIGHT,
    "model": rlbc.LEGION_MODEL,
    "seed": getattr(rlbc, 'LEGION_SEED', -1),
    "refiner_control_percentage": getattr(rlbc, 'LEGION_REFINER_CONTROL_PERCENTAGE', 0.6),
    "refiner_method": getattr(rlbc, 'LEGION_REFINER_METHOD', 'Post-Apply (Normal)'),
    "refiner_upscale": getattr(rlbc, 'LEGION_REFINER_UPSCALE', 2),
    "refiner_upscale_method": getattr(rlbc, 'LEGION_REFINER_UPSCALE_METHOD', '4xRealWebPhoto_v4_dat2.safetensors'),
    "refiner_steps": getattr(rlbc, 'LEGION_REFINER_STEPS', 7),
    "automatic_vae": getattr(rlbc, 'LEGION_AUTOMATIC_VAE', True),
    "loras": loras_as_strings,
}

# Cleaned fallback: no loras and no refiner keys
fallback_clean = dict(fallback_full)
fallback_clean.pop("loras", None)
for k in ["refiner_control_percentage", "refiner_method", "refiner_upscale", "refiner_upscale_method", "refiner_steps"]:
    fallback_clean.pop(k, None)

# Progressive retry strategy per endpoint
saved = False
for endpoint in ENDPOINTS:
    print(f"Trying endpoint: {endpoint} with cleaned fallback payload (no LoRAs/refiner)")
    try:
        r = requests.post(endpoint, json=fallback_clean, timeout=120)
    except Exception as e:
        print("Request error:", e)
        continue

    text = (getattr(r, 'text', '') or '')
    print(f"Status {r.status_code}; body (first 300 chars): {text[:300]!r}")

    # Check JSON images
    try:
        data = r.json()
    except Exception:
        data = None

    if r.ok and data and data.get("images"):
        first = data["images"][0]
        print("Got images from cleaned fallback")
        image_bytes = None
        if isinstance(first, str) and first.startswith("data:"):
            b64 = first.split(",", 1)[1]
            image_bytes = base64.b64decode(b64)
        else:
            image_bytes = None
            # attempt download
            try:
                image_bytes = requests.get(f"{URL}/{first}", timeout=60).content
            except Exception as e:
                print("Failed to download image path:", e)
        if image_bytes:
            with open("swarmui_fallback_clean.png", "wb") as f:
                f.write(image_bytes)
            print("Saved swarmui_fallback_clean.png")
            saved = True
            break

    # If no success, analyze text to decide next step
    low = text.lower()
    # if server complains about missing model, try full fallback
    if "no model input given" in low or "invalid model" in low or "are you sure that model name is correct" in low:
        print("Server indicates missing/invalid model; trying full fallback payload (with model and defaults)")
        try:
            r2 = requests.post(endpoint, json=fallback_full, timeout=120)
            print(f"Fallback status {r2.status_code}; body: {(getattr(r2,'text','') or '')[:300]!r}")
            try:
                data2 = r2.json()
            except Exception:
                data2 = None
            if r2.ok and data2 and data2.get("images"):
                first = data2["images"][0]
                image_bytes = None
                if isinstance(first, str) and first.startswith("data:"):
                    image_bytes = base64.b64decode(first.split(",", 1)[1])
                else:
                    try:
                        image_bytes = requests.get(f"{URL}/{first}", timeout=60).content
                    except Exception as e:
                        print("Failed to download image path after fallback:", e)
                if image_bytes:
                    with open("swarmui_fallback_full.png", "wb") as f:
                        f.write(image_bytes)
                    print("Saved swarmui_fallback_full.png")
                    saved = True
                    break

            # If fallback failed and mentions loras or refiner, try removing them progressively
            low2 = (getattr(r2,'text','') or '').lower()
            if "loras" in low2 or "lora" in low2:
                print("Fallback rejected LoRAs; retrying without LoRAs")
                cleaned = dict(fallback_full)
                cleaned.pop("loras", None)
                r3 = requests.post(endpoint, json=cleaned, timeout=120)
                print(f"Retry no LORAs status {r3.status_code}; body: {(getattr(r3,'text','') or '')[:300]!r}")
                try:
                    d3 = r3.json()
                except Exception:
                    d3 = None
                if r3.ok and d3 and d3.get("images"):
                    first = d3["images"][0]
                    if isinstance(first, str) and first.startswith("data:"):
                        img = base64.b64decode(first.split(",", 1)[1])
                        with open("swarmui_fallback_no_loras.png", "wb") as f:
                            f.write(img)
                        print("Saved swarmui_fallback_no_loras.png")
                        saved = True
                        break

            if "refiner" in low2 or "unrecognized" in low2:
                print("Fallback rejected refiner; retrying without refiner keys")
                cleaned = dict(fallback_full)
                for k in ["refiner_control_percentage","refiner_method","refiner_upscale","refiner_upscale_method","refiner_steps"]:
                    cleaned.pop(k, None)
                r4 = requests.post(endpoint, json=cleaned, timeout=120)
                print(f"Retry no refiner status {r4.status_code}; body: {(getattr(r4,'text','') or '')[:300]!r}")
                try:
                    d4 = r4.json()
                except Exception:
                    d4 = None
                if r4.ok and d4 and d4.get("images"):
                    first = d4["images"][0]
                    if isinstance(first, str) and first.startswith("data:"):
                        img = base64.b64decode(first.split(",", 1)[1])
                        with open("swarmui_fallback_no_refiner.png", "wb") as f:
                            f.write(img)
                        print("Saved swarmui_fallback_no_refiner.png")
                        saved = True
                        break

        except Exception as e:
            print("Error during fallback attempt:", e)

    # If we're here and not saved, continue to next endpoint
    print("Moving to next endpoint")

if not saved:
    print("No image could be produced by any strategy")
else:
    print("Done")
