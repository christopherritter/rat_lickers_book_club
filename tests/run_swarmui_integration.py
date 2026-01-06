import sys
import os
# Ensure project root is on sys.path so we can import local module
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rlbc_daily_to_notion import generate_image_on_legion

print("Starting live SwarmUI integration test...")
img = generate_image_on_legion("Live Test Title", "Integration test description")
if img:
    out = "swarmui_live_test.png"
    with open(out, "wb") as f:
        f.write(img)
    print("Image saved to:", out)
else:
    print("No image returned")
