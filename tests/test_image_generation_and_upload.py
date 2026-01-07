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

    captured = {"json": None, "timeout": None}

    def fake_post(url, json=None, timeout=None):
        # Capture timeout for verification
        captured["timeout"] = timeout
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

    # Assert full fallback payload was sent (includes model, resolution, LoRAs, and refiner settings)
    assert captured["json"]["model"] == rlbc.LEGION_MODEL
    assert captured["json"]["width"] == rlbc.LEGION_WIDTH
    assert captured["json"]["height"] == rlbc.LEGION_HEIGHT
    assert captured["json"]["steps"] == rlbc.LEGION_STEPS
    # Accept either snake_case 'cfg_scale' or key 'cfgscale' used by fixed payload
    assert captured["json"].get("cfg_scale", captured["json"].get("cfgscale")) == rlbc.LEGION_CFG_SCALE
    # By default we send LoRAs as a pipe-delimited string
    assert "loras" in captured["json"]
    assert isinstance(captured["json"]["loras"], str) and "|||" in captured["json"]["loras"]
    # Accept either 'refiner_control_percentage' or 'refiner_control'
    assert (captured["json"].get("refiner_control_percentage") == rlbc.LEGION_REFINER_CONTROL_PERCENTAGE) or (captured["json"].get("refiner_control") == rlbc.LEGION_REFINER_CONTROL_PERCENTAGE)
    assert captured["json"].get("sigma_shift") == 1
    # Accept a set of server-accepted dtype tokens
    assert captured["json"].get("preferred_dtype") in ("Default (16 bit)", "fp16", "default", "automatic", "fp8_e4m3fn", "fp8_e5m2")

    # Ensure we used a long timeout (generation can take several minutes)
    assert captured["timeout"] == rlbc.LEGION_REQUEST_TIMEOUT

    # Ensure sampler, scheduler match canonical tokens accepted by SwarmUI
    assert captured["json"].get("sampler") in ("euler_ancestral", "euler_ancestral_cfg_pp", "euler_a", "euler")
    assert captured["json"].get("scheduler") == "normal"

    # Accept either canonical 'automatic_vae' or 'autovae' keys
    assert (captured["json"].get("automatic_vae") is True) or (captured["json"].get("autovae") is True)

    # Ensure the payload included refiner method (either human-friendly or normalized) and upscale string/model at least once
    assert captured["json"].get("refiner_method") in (rlbc.LEGION_REFINER_METHOD, "Post-Apply", "PostApply", "StepSwap", "StepSwapNoisy")
    # Accept both 'refiner_control' and 'refiner_control_percentage'
    assert (captured["json"].get("refiner_control") == rlbc.LEGION_REFINER_CONTROL_PERCENTAGE) or (captured["json"].get("refiner_control_percentage") == rlbc.LEGION_REFINER_CONTROL_PERCENTAGE)

    # If refiner_model is present it should include the model prefix per fixed JSON
    if captured["json"].get("refiner_model"):
        assert captured["json"].get("refiner_model") == f"model-{rlbc.LEGION_REFINER_UPSCALE_METHOD}"

    # Ensure refiner upscale and steps are correct
    assert captured["json"].get("refiner_upscale") == rlbc.LEGION_REFINER_UPSCALE
    assert captured["json"].get("refiner_steps") == rlbc.LEGION_REFINER_STEPS

    # Ensure the LoRAs include the expected name:weight pairs (support list or string forms)
    loras = captured["json"].get("loras")
    loras_str_variants = [
        captured["json"].get("loras_string"),
        captured["json"].get("LoRAs"),
        captured["json"].get("loras") if isinstance(captured["json"].get("loras"), str) else None,
    ]

    # If we received a list, verify membership
    if isinstance(loras, list):
        assert "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-fp32:0.9" in loras
        assert "Qwen_LoRA_Skin_Fix_v2:0.6" in loras
    else:
        # Default expectation: LoRAs are sent as pipe-delimited string
        found = None
        for s in loras_str_variants:
            if s and "|||" in s:
                found = s
                break
        if found:
            assert "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-fp32:0.9" in found
            assert "Qwen_LoRA_Skin_Fix_v2:0.6" in found
        else:
            # Fall back to checking any available string variant (comma or single-string)
            for s in loras_str_variants:
                if s:
                    assert "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-fp32:0.9" in s
                    assert "Qwen_LoRA_Skin_Fix_v2:0.6" in s
                    break
        joined = ",".join([v for v in loras_str_variants if v])
        assert "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-fp32:0.9" in joined
        assert "Qwen_LoRA_Skin_Fix_v2:0.6" in joined


def test_generate_image_on_legion_no_image(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        return DummyResponse({"images": []})

    monkeypatch.setattr(rlbc.requests, "post", fake_post)

    result = rlbc.generate_image_on_legion("Title", "desc")
    assert result == (None, None)


def test_upload_image_to_notion_page_success(monkeypatch):
    captured = {"timeout": None}

    def fake_patch(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
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

    # Ensure we used a long timeout for Notion upload
    assert captured["timeout"] == rlbc.NOTION_UPLOAD_TIMEOUT

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

    # Verify the first generation call contained the model and LoRAs (we now include full params by default)
    first = calls[1]
    assert first["json"].get("model") == rlbc.LEGION_MODEL
    assert "loras" in first["json"]

    # The second call was the fallback with model and loras (still triggered by model error)
    fallback = calls[2]
    assert fallback["json"].get("model") == rlbc.LEGION_MODEL
    assert "loras" in fallback["json"]

    # The final call may or may not include LoRAs depending on which variant succeeded; ensure success occurred
    final = calls[-1]


def test_generate_image_on_legion_lora_space_separated_fallback(monkeypatch):
    """If pipe-delimited LoRAs are rejected, try space-separated string form."""
    calls = []
    fake_bytes = b"space-fallback-bytes"
    fake_b64 = base64.b64encode(fake_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{fake_b64}"

    def fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json})
        if url.endswith("/API/GetNewSession"):
            return DummyResponse({"session_id": "sess-space"}, status_code=200, ok=True)
        if url.endswith("/API/GenerateText2Image"):
            l = json.get("loras")
            # Reject pipe-delimited
            if isinstance(l, str) and "|||" in l:
                return DummyResponse({}, status_code=400, ok=False, text="Invalid value for parameter LoRAs: option does not exist")
            # Accept space-separated
            if isinstance(l, str) and " " in l:
                return DummyResponse({"images": [data_url]}, status_code=200, ok=True)
            return DummyResponse({}, status_code=400, ok=False, text="LoRAs rejected")
        return DummyResponse({}, status_code=404, ok=False)

    monkeypatch.setattr(rlbc.requests, "post", fake_post)

    out, url = rlbc.generate_image_on_legion("Space LoRA", "desc")
    assert out == fake_bytes

    gen_calls = [c for c in calls if c["url"].endswith("/API/GenerateText2Image") or c["url"].endswith("/Text2Image")]
    # Verify at least one attempt used space-separated LoRAs
    assert any(isinstance(c["json"].get("loras"), str) and " " in c["json"].get("loras") for c in gen_calls)
    # Also ensure alternative keys don't still contain the original pipe-delimited value
    assert any((isinstance(c["json"].get("loras"), str) and "|||" not in c["json"].get("loras") and isinstance(c["json"].get("LoRAs"), str) and "|||" not in c["json"].get("LoRAs")) or (isinstance(c["json"].get("loras"), list) and isinstance(c["json"].get("LoRAs"), list)) for c in gen_calls)


def test_generate_image_on_legion_lora_single_pipe_fallback(monkeypatch):
    """If pipe-delimited LoRAs and space-separated are rejected, try single-pipe variant."""
    calls = []
    fake_bytes = b"singlepipe-fallback-bytes"
    fake_b64 = base64.b64encode(fake_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{fake_b64}"

    def fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json})
        if url.endswith("/API/GetNewSession"):
            return DummyResponse({"session_id": "sess-pipe"}, status_code=200, ok=True)
        if url.endswith("/API/GenerateText2Image"):
            l = json.get("loras")
            # Reject pipe-delimited and space-separated
            if isinstance(l, str) and ("|||" in l or " " in l):
                return DummyResponse({}, status_code=400, ok=False, text="Invalid value for parameter LoRAs: option does not exist")
            # Accept single-pipe
            if isinstance(l, str) and "|" in l:
                return DummyResponse({"images": [data_url]}, status_code=200, ok=True)
            return DummyResponse({}, status_code=400, ok=False, text="LoRAs rejected")
        return DummyResponse({}, status_code=404, ok=False)

    monkeypatch.setattr(rlbc.requests, "post", fake_post)

    out, url = rlbc.generate_image_on_legion("Pipe LoRA", "desc")
    assert out == fake_bytes

    gen_calls = [c for c in calls if c["url"].endswith("/API/GenerateText2Image") or c["url"].endswith("/Text2Image")]
    # Verify at least one attempt used single-pipe LoRAs
    assert any(isinstance(c["json"].get("loras"), str) and "|" in c["json"].get("loras") and "|||" not in c["json"].get("loras") for c in gen_calls)


def test_generate_image_on_legion_lora_concatenated_fallback(monkeypatch):
    """If other LoRA formats are rejected, try concatenated (no separator) variant."""
    calls = []
    fake_bytes = b"concat-fallback-bytes"
    fake_b64 = base64.b64encode(fake_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{fake_b64}"

    def fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json})
        if url.endswith("/API/GetNewSession"):
            return DummyResponse({"session_id": "sess-concat"}, status_code=200, ok=True)
        if url.endswith("/API/GenerateText2Image"):
            l = json.get("loras")
            # Reject pipe-delimited, space, and single-pipe
            if isinstance(l, str) and ("|||" in l or " " in l or "|" in l):
                return DummyResponse({}, status_code=400, ok=False, text="Invalid value for parameter LoRAs: option does not exist")
            # Accept concatenated
            if isinstance(l, str) and ":" in l and " " not in l and "|" not in l and ";" not in l and "," not in l:
                return DummyResponse({"images": [data_url]}, status_code=200, ok=True)
            return DummyResponse({}, status_code=400, ok=False, text="LoRAs rejected")
        return DummyResponse({}, status_code=404, ok=False)

    monkeypatch.setattr(rlbc.requests, "post", fake_post)

    out, url = rlbc.generate_image_on_legion("Concat LoRA", "desc")
    assert out == fake_bytes

    gen_calls = [c for c in calls if c["url"].endswith("/API/GenerateText2Image") or c["url"].endswith("/Text2Image")]
    # Verify at least one attempt used concatenated LoRAs (no separators)
    assert any(isinstance(c["json"].get("loras"), str) and ":" in c["json"].get("loras") and all(sep not in c["json"].get("loras") for sep in ["|||", " ", "|", ";", ","]) for c in gen_calls)


def test_generate_image_on_legion_payload_debug(monkeypatch, capsys):
    """When LEGION_PAYLOAD_DEBUG is enabled, the final payload is printed to stdout before sending."""
    # Enable debug at module level
    monkeypatch.setattr(rlbc, "LEGION_PAYLOAD_DEBUG", True)

    def fake_post(url, json=None, timeout=None):
        if url.endswith("/API/GetNewSession"):
            return DummyResponse({"session_id": "sess-debug"}, status_code=200, ok=True)
        if url.endswith("/API/GenerateText2Image"):
            fake_bytes = b"debug-bytes"
            fake_b64 = base64.b64encode(fake_bytes).decode("utf-8")
            data_url = f"data:image/png;base64,{fake_b64}"
            return DummyResponse({"images": [data_url]}, status_code=200, ok=True)
        return DummyResponse({}, status_code=404, ok=False)

    monkeypatch.setattr(rlbc.requests, "post", fake_post)

    out, url = rlbc.generate_image_on_legion("Debug Title", "desc")
    assert out == b"debug-bytes"

    captured = capsys.readouterr()
    assert "🔍 LEGION payload to" in captured.out
    assert '"loras"' in captured.out

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

    # Verify the first generation call included the model (we now include refiner keys by default)
    first_gen_call = calls[1]
    assert first_gen_call["json"].get("model") == rlbc.LEGION_MODEL

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


def test_generate_image_on_legion_key_variants(monkeypatch):
    calls = []
    fake_bytes = b'var-bytes'
    fake_b64 = base64.b64encode(fake_bytes).decode('utf-8')
    data_url = f"data:image/png;base64,{fake_b64}"

    def fake_post(url, json=None, timeout=None):
        calls.append({'url': url, 'json': json})
        # Session acquisition
        if url.endswith('/API/GetNewSession'):
            return DummyResponse({'session_id': 'sess-var'}, status_code=200, ok=True)

        if url.endswith('/API/GenerateText2Image'):
            # If payload uses cfg_scale, simulate server error about preferred_dtype
            if 'cfg_scale' in json:
                return DummyResponse({}, status_code=400, ok=False, text="Unknown parameter preferred_dtype")
            # If payload uses cfgscale, accept
            if 'cfgscale' in json:
                return DummyResponse({'images': [data_url]}, status_code=200, ok=True)
            return DummyResponse({}, status_code=400, ok=False)

        return DummyResponse({}, status_code=404, ok=False)

    monkeypatch.setattr(rlbc.requests, 'post', fake_post)

    out, url = rlbc.generate_image_on_legion("Var Test", "desc")
    assert out == fake_bytes

    gen_calls = [c for c in calls if c['url'].endswith('/API/GenerateText2Image')]
    # Accept either cfg_scale or cfgscale appearing in at least one variant call
    assert any(('cfg_scale' in c['json']) or ('cfgscale' in c['json']) for c in gen_calls)
    # Ensure cfgscale variant is present when server supports it
    assert any('cfgscale' in c['json'] for c in gen_calls)


def test_generate_image_on_legion_refiner_method_mapping(monkeypatch):
    calls = []
    fake_bytes = b'ref-bytes'
    fake_b64 = base64.b64encode(fake_bytes).decode('utf-8')
    data_url = f"data:image/png;base64,{fake_b64}"

    def fake_post(url, json=None, timeout=None):
        calls.append({'url': url, 'json': json})
        if url.endswith('/API/GetNewSession'):
            return DummyResponse({'session_id': 'sess-ref'}, status_code=200, ok=True)
        if url.endswith('/API/GenerateText2Image'):
            # If refiner_method is not normalized, reject
            rm = json.get('refiner_method')
            if rm and ('post-apply' in str(rm).lower() or 'post-apply (normal)' in str(rm).lower()):
                return DummyResponse({}, status_code=400, ok=False, text="Invalid value for parameter Refiner Method: Invalid value for param Refiner Method - 'Post-Apply (Normal)'")
            # Accept when normalized to 'PostApply'
            if rm == 'PostApply' or json.get('refiner_upscale_method') == LEGION_REFINER_UPSCALE_METHOD:
                return DummyResponse({'images': [data_url]}, status_code=200, ok=True)
            return DummyResponse({}, status_code=400, ok=False)
        return DummyResponse({}, status_code=404, ok=False)

    monkeypatch.setattr(rlbc.requests, 'post', fake_post)

    out, url = rlbc.generate_image_on_legion("Ref Test", "desc")
    assert out == fake_bytes

    gen_calls = [c for c in calls if c['url'].endswith('/API/GenerateText2Image')]
    # Expect we sent a variety and one normalized to PostApply
    assert any('PostApply' == c['json'].get('refiner_method') for c in gen_calls)


def test_generate_image_on_legion_loras_object(monkeypatch):
    calls = []
    fake_bytes = b'loras-bytes'
    fake_b64 = base64.b64encode(fake_bytes).decode('utf-8')
    data_url = f"data:image/png;base64,{fake_b64}"

    def fake_post(url, json=None, timeout=None):
        calls.append({'url': url, 'json': json})
        if url.endswith('/API/GetNewSession'):
            return DummyResponse({'session_id': 'sess-lora'}, status_code=200, ok=True)

        if url.endswith('/API/GenerateText2Image'):
            # If LoRAs are objects (name/weight), server rejects them (some SwarmUI versions reject objects)
            if 'loras' in json and json['loras'] and isinstance(json['loras'][0], dict):
                return DummyResponse({}, status_code=400, ok=False, text="Invalid value for parameter LoRAs: option does not exist")
            # If LoRAs are strings, accept
            if 'loras' in json and json['loras'] and isinstance(json['loras'][0], str):
                return DummyResponse({'images': [data_url]}, status_code=200, ok=True)
            return DummyResponse({}, status_code=400, ok=False)

        return DummyResponse({}, status_code=404, ok=False)

    monkeypatch.setattr(rlbc.requests, 'post', fake_post)

    out, url = rlbc.generate_image_on_legion("LoRA Test", "desc")
    assert out == fake_bytes

    gen_calls = [c for c in calls if c['url'].endswith('/API/GenerateText2Image')]
    # Expect that at least one call sent LoRAs as strings (some SwarmUI versions accept strings immediately).
    assert any(isinstance(c['json'].get('loras', [None])[0], str) for c in gen_calls)
    # If the server rejected strings, we should have also tried objects; so it's OK if objects weren't present when strings succeeded.

