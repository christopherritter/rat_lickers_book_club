import base64
import json
import requests
import rlbc_daily_to_notion as rlbc
import pytest


class DummyResponse:
    def __init__(self, json_data=None, status_code=200, ok=True, text=""):
        self._json = json_data or {}
        self.status_code = status_code
        self.ok = ok
        self.text = text

    def json(self):
        return self._json

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def test_generate_image_on_legion_success(monkeypatch):
    fake_bytes = b"fake-image-bytes"
    fake_b64 = base64.b64encode(fake_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{fake_b64}"

    captured = {"json": None}

    def fake_post(url, json=None, timeout=None):
        # Session acquisition
        if url.endswith("/API/GetNewSession"):
            return DummyResponse({"session_id": "sess-1"}, status_code=200, ok=True)
        # Simulate LEGION return payload (data URL)
        if url.endswith("/API/GenerateText2Image"):
            captured["json"] = json
            return DummyResponse({"images": [data_url]}, status_code=200, ok=True)
        return DummyResponse({}, status_code=404, ok=False)

    monkeypatch.setattr(rlbc.requests, "post", fake_post)

    result, url = rlbc.generate_image_on_legion("Short Title", "A little description")
    assert result == fake_bytes
    assert url is None

    # Assert cleaned fallback payload was used (model & resolution present, but no LoRAs/refiner)
    assert captured["json"]["model"] == rlbc.LEGION_MODEL
    assert captured["json"]["width"] == rlbc.LEGION_WIDTH
    assert captured["json"]["height"] == rlbc.LEGION_HEIGHT
    assert captured["json"]["steps"] == rlbc.LEGION_STEPS
    assert captured["json"]["cfg_scale"] == rlbc.LEGION_CFG_SCALE
    assert "loras" not in captured["json"]
    assert "refiner_control_percentage" not in captured["json"]


def test_generate_image_on_legion_no_image(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        return DummyResponse({"images": []})

    monkeypatch.setattr(rlbc.requests, "post", fake_post)

    result = rlbc.generate_image_on_legion("Title", "desc")
    assert result == (None, None)


def test_upload_image_to_notion_page_success(monkeypatch):
    captured = {}

    def fake_patch(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return DummyResponse({}, status_code=200, ok=True)

    monkeypatch.setattr(rlbc.requests, "patch", fake_patch)

    # Ensure NOTION_API_KEY is set for header formatting clarity
    monkeypatch.setattr(rlbc, "NOTION_API_KEY", "fake-token")

    # Test embedding bytes as data URL
    ok = rlbc.upload_image_to_notion_page("page-123", b"abc", caption="caption")
    assert ok is True

    assert captured["url"].endswith("/page-123/children")
    children = captured["json"]["children"]
    assert children and children[0]["type"] == "image"

    data_url = children[0]["image"]["external"]["url"]
    assert data_url.startswith("data:image/png;base64,")

    encoded = data_url.split(",", 1)[1]
    assert base64.b64decode(encoded) == b"abc"

    # Test using external_url (avoid embedding)
    captured.clear()
    ok = rlbc.upload_image_to_notion_page("page-123", None, caption="caption", external_url="https://example.com/img.png")
    assert ok is True
    children = captured["json"]["children"]
    assert children[0]["image"]["external"]["url"] == "https://example.com/img.png"


def test_create_notion_page_from_post_book_feature(monkeypatch):
    # Capture calls and payloads
    captured = {"patched": None, "posted": []}

    fake_img_bytes = b"img123"
    fake_img_b64 = base64.b64encode(fake_img_bytes).decode("utf-8")

    def fake_post(url, headers=None, json=None, timeout=None):
        # Session acquisition
        if url.endswith("/API/GetNewSession"):
            return DummyResponse({"session_id": "sess-xyz"}, status_code=200, ok=True)

        # LEGION image generation endpoint (return image path)
        if url.startswith(rlbc.LEGION_SWARMUI_URL) and url.endswith("/GenerateText2Image"):
            return DummyResponse({"images": ["View/local/raw/2026-01-06/image.png"]})

        # Notion page creation
        if url == rlbc.NOTION_PAGES_URL:
            # Return a fake page object
            return DummyResponse({"id": "page-xyz"})

        return DummyResponse({}, status_code=404, ok=False)

    def fake_get(url, timeout=None):
        class R:
            def __init__(self):
                self.content = fake_img_bytes
            def raise_for_status(self):
                return None
        return R()

    def fake_patch(url, headers=None, json=None, timeout=None):
        captured["patched"] = {"url": url, "headers": headers, "json": json}
        return DummyResponse({}, status_code=200, ok=True)

    monkeypatch.setattr(rlbc.requests, "post", fake_post)
    monkeypatch.setattr(rlbc.requests, "get", fake_get)
    monkeypatch.setattr(rlbc.requests, "patch", fake_patch)

    # Ensure necessary module-level config exists
    monkeypatch.setattr(rlbc, "NOTION_API_KEY", "fake-token")
    monkeypatch.setattr(rlbc, "RLBC_DATABASE_ID", "db-123")
    monkeypatch.setattr(rlbc, "RLBC_TITLE_PROPERTY", "Name")

    post = {
        "name": "Feature Post",
        "date": "2026-01-06",
        "post_type": "Book Feature",
        "platform": "Facebook",
        "book_title": "Short Title",
        "book_description": "A satirical short description",
    }

    page = rlbc.create_notion_page_from_post(post)

    assert page["id"] == "page-xyz"

    # Verify image patch payload used the external SwarmUI URL
    assert captured["patched"] is not None
    payload = captured["patched"]["json"]
    children = payload["children"]
    assert children and children[0]["type"] == "image"

    data_url = children[0]["image"]["external"]["url"]
    assert data_url == f"{rlbc.LEGION_SWARMUI_URL}/View/local/raw/2026-01-06/image.png"


def test_get_swarmui_session_success(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        return DummyResponse({"session_id": "sess-abc123"})

    monkeypatch.setattr(rlbc.requests, "post", fake_post)
    sess = rlbc.get_swarmui_session()
    assert sess == "sess-abc123"


def test_download_image_from_swarmui(monkeypatch):
    def fake_get(url, timeout=None):
        return DummyResponse({}, ok=True, status_code=200, text="",)

    # Return the bytes via response.content
    class Resp:
        def __init__(self, content):
            self.content = content
        def raise_for_status(self):
            return None

    def fake_get_bytes(url, timeout=None):
        return Resp(b"downloaded-bytes")

    monkeypatch.setattr(rlbc.requests, "get", fake_get_bytes)
    out = rlbc.download_image_from_swarmui("View/local/raw/2026-01-01/image.png")
    assert out == b"downloaded-bytes"


def test_generate_image_on_legion_fallbacks(monkeypatch):
    # Simulate first endpoint returning 500, second returning image path, and download returns bytes
    called = {"posts": []}

    def fake_post(url, json=None, timeout=None):
        called["posts"].append(url)
        if url.endswith("/API/GenerateText2Image"):
            return DummyResponse({}, status_code=500, ok=False)
        if url.endswith("/Text2Image"):
            return DummyResponse({"images": ["View/local/raw/2026-01-01/fallback.png"]}, status_code=200, ok=True)
        # Default: session acquisition
        if url.endswith("/API/GetNewSession"):
            return DummyResponse({"session_id": "sess-fallback"}, status_code=200, ok=True)
        return DummyResponse({}, status_code=404, ok=False)

    def fake_get(url, timeout=None):
        # Simulate downloading the file from SwarmUI
        class R:
            def __init__(self):
                self.content = b"final-bytes"
            def raise_for_status(self):
                return None
        return R()

    monkeypatch.setattr(rlbc.requests, "post", fake_post)
    monkeypatch.setattr(rlbc.requests, "get", fake_get)

    out, url = rlbc.generate_image_on_legion("Fallback Title", "desc")
    assert out == b"final-bytes"
    assert url == f"{rlbc.LEGION_SWARMUI_URL}/View/local/raw/2026-01-01/fallback.png"
    # Ensure it tried the primary then fallback
    assert any(p.endswith("/API/GenerateText2Image") for p in called["posts"])
    assert any(p.endswith("/Text2Image") for p in called["posts"])


def test_generate_image_on_legion_retries_without_loras(monkeypatch):
    calls = []
    fake_bytes = b"bytes-after-retry"
    fake_b64 = base64.b64encode(fake_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{fake_b64}"

    def fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json})
        # session
        if url.endswith("/API/GetNewSession"):
            return DummyResponse({"session_id": "sess-retry"}, status_code=200, ok=True)
        # first attempt returns model error so we retry with fallback (which contains loras)
        if len(calls) == 2:
            return DummyResponse({}, status_code=400, ok=False, text="Invalid value for parameter Model: Invalid model value")
        # fallback attempt includes loras and this server rejects LoRAs
        if len(calls) == 3:
            return DummyResponse({}, status_code=400, ok=False, text="Invalid value for parameter LoRAs: option does not exist")
        # final retry succeeds (no loras)
        if url.endswith("/API/GenerateText2Image") and len(calls) >= 4:
            return DummyResponse({"images": [data_url]}, status_code=200, ok=True)
        return DummyResponse({}, status_code=404, ok=False)

    monkeypatch.setattr(rlbc.requests, "post", fake_post)

    out, url = rlbc.generate_image_on_legion("Retry LoRA", "desc")
    assert out == fake_bytes
    assert url is None

    # Verify the first generation call used cleaned fallback payload (has model, no loras)
    first = calls[1]
    assert first["json"].get("model") == rlbc.LEGION_MODEL
    assert "loras" not in first["json"]

    # The second call was the fallback with model and loras (triggered by model error)
    fallback = calls[2]
    assert fallback["json"].get("model") == rlbc.LEGION_MODEL
    assert "loras" in fallback["json"]

    # Final call should not contain loras
    final = calls[-1]
    assert "loras" not in final["json"]

def test_generate_image_on_legion_handles_no_model_input(monkeypatch):
    calls = []
    fake_bytes = b"bytes-from-no-model-retry"
    fake_b64 = base64.b64encode(fake_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{fake_b64}"

    def fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json})
        # session
        if url.endswith("/API/GetNewSession"):
            return DummyResponse({"session_id": "sess-nomodel"}, status_code=200, ok=True)
        # first generation attempt: server complains that no model is set
        if len(calls) == 2:
            return DummyResponse({}, status_code=400, ok=False, text="No model input given. Did your UI load properly?")
        # fallback attempt should include model and succeed
        if url.endswith("/API/GenerateText2Image") and len(calls) >= 3:
            return DummyResponse({"images": [data_url]}, status_code=200, ok=True)
        return DummyResponse({}, status_code=404, ok=False)

    monkeypatch.setattr(rlbc.requests, "post", fake_post)

    out, url = rlbc.generate_image_on_legion("No Model", "desc")
    assert out == fake_bytes
    assert url is None

    # Verify the fallback call included the model field
    fallback_call = next((c for c in calls if c["json"].get("model")), None)
    assert fallback_call is not None and fallback_call["json"]["model"] == rlbc.LEGION_MODEL

def test_generate_image_on_legion_retries_without_refiner(monkeypatch):
    calls = []
    fake_bytes = b"bytes-refiner-removed"
    fake_b64 = base64.b64encode(fake_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{fake_b64}"

    def fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json})
        # session
        if url.endswith("/API/GetNewSession"):
            return DummyResponse({"session_id": "sess-ref"}, status_code=200, ok=True)
        # first attempt rejects refiner (we expect minimal payload to be sent initially, but for this test we will simulate the server returning a refiner error when we send the fallback payload)
        if len(calls) == 3:
            return DummyResponse({}, status_code=400, ok=False, text="request parameter 'refiner' is unrecognized, skipping")
        # retry succeeds (accept success on either endpoint for the retry)
        if (url.endswith("/API/GenerateText2Image") or url.endswith("/Text2Image")) and len(calls) >= 4:
            return DummyResponse({"images": [data_url]}, status_code=200, ok=True)
        return DummyResponse({}, status_code=404, ok=False)

    monkeypatch.setattr(rlbc.requests, "post", fake_post)

    out, url = rlbc.generate_image_on_legion("Retry Refiner", "desc")
    assert out == fake_bytes
    assert url is None

    # Verify the first generation call used cleaned fallback payload (has model, no refiner)
    first_gen_call = calls[1]
    assert first_gen_call["json"].get("model") == rlbc.LEGION_MODEL
    assert "refiner_control_percentage" not in first_gen_call["json"]

    # Depending on which endpoint returned which error, a full fallback payload may or may not have been sent; what's important is the final success without refiner keys.

    # The final retry call should not include refiner keys
    final_call = calls[-1]
    assert "refiner_control_percentage" not in final_call["json"]


    """End-to-end style test: generate a Book Feature post and image, create Notion DB row, upload image."""
    recorded = {"post_payload": None, "patch_payload": None}

    fake_img = b"final-image-bytes"
    fake_img_b64 = base64.b64encode(fake_img).decode("utf-8")

    def fake_post(url, headers=None, json=None, timeout=None):
        # Session acquisition
        if url.endswith("/API/GetNewSession"):
            return DummyResponse({"session_id": "sess-final"}, status_code=200, ok=True)

        # LEGION image generation call (return data URL)
        if url.startswith(rlbc.LEGION_SWARMUI_URL) and url.endswith("/GenerateText2Image"):
            data_url = f"data:image/png;base64,{fake_img_b64}"
            return DummyResponse({"images": [data_url]})

        # Notion creates a new page (row in DB)
        if url == rlbc.NOTION_PAGES_URL:
            recorded["post_payload"] = {"url": url, "headers": headers, "json": json}
            # Return a Notion page id as the created row
            return DummyResponse({"id": "page-final"})

        return DummyResponse({}, status_code=404, ok=False)

    def fake_patch(url, headers=None, json=None, timeout=None):
        recorded["patch_payload"] = {"url": url, "headers": headers, "json": json}
        return DummyResponse({}, status_code=200, ok=True)

    monkeypatch.setattr(rlbc.requests, "post", fake_post)
    monkeypatch.setattr(rlbc.requests, "patch", fake_patch)

    monkeypatch.setattr(rlbc, "NOTION_API_KEY", "fake-token")
    monkeypatch.setattr(rlbc, "RLBC_DATABASE_ID", "db-final")
    monkeypatch.setattr(rlbc, "RLBC_TITLE_PROPERTY", "Name")

    # Use minimal post so format_post_text will generate default book feature body
    post = {
        "name": "Final Feature",
        "date": "2026-01-06",
        "post_type": "Book Feature",
        "platform": "Facebook",
        "book_title": "Final Title",
        "book_description": "Description here",
    }

    page = rlbc.create_notion_page_from_post(post)

    # Verify created page is returned
    assert page["id"] == "page-final"

    # Verify the POST to Notion contained parent database id and properties
    assert recorded["post_payload"] is not None
    pp = recorded["post_payload"]["json"]
    assert pp["parent"]["type"] == "database_id"
    assert pp["parent"]["database_id"] == "db-final"

    # Check title property populated
    assert "Name" in pp["properties"]
    assert pp["properties"]["Name"]["title"][0]["text"]["content"] == "Final Feature"

    # Confirm children include a heading block for the Book Feature
    children = pp["children"]
    assert any(c["type"] == "heading_3" and "This Week's Selection" in c["heading_3"]["rich_text"][0]["text"]["content"] for c in children)

    # Verify that the image upload PATCH targeted the new page and contained the data URL
    assert recorded["patch_payload"] is not None
    patch_json = recorded["patch_payload"]["json"]
    patch_children = patch_json["children"]
    assert patch_children and patch_children[0]["type"] == "image"
    uploaded_data_url = patch_children[0]["image"]["external"]["url"]
    # The base64 part should decode to our fake image bytes
    assert base64.b64decode(uploaded_data_url.split(",", 1)[1]) == fake_img

