import os
import sys
import traceback

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rlbc_daily_to_notion import create_notion_page_from_post

print("Starting live end-to-end test: generate Book Feature + upload to Notion")

NOTION_API_KEY = os.getenv("NOTION_API_KEY")
RLBC_DATABASE_ID = os.getenv("RLBC_DATABASE_ID")

if not NOTION_API_KEY or not RLBC_DATABASE_ID:
    print("Missing Notion credentials (NOTION_API_KEY or RLBC_DATABASE_ID). Aborting end-to-end test.")
    print(f"NOTION_API_KEY={'SET' if NOTION_API_KEY else 'MISSING'}, RLBC_DATABASE_ID={'SET' if RLBC_DATABASE_ID else 'MISSING'})")
    sys.exit(0)

post = {
    "name": "Live E2E Test",
    "date": None,
    "post_type": "Book Feature",
    "platform": "Facebook",
    "book_title": "E2E Title",
    "book_description": "Testing live generate and upload",
}

try:
    page = create_notion_page_from_post(post)
    print("Page created:", page.get("id"))
except Exception as e:
    print("Error during end-to-end run:")
    traceback.print_exc()