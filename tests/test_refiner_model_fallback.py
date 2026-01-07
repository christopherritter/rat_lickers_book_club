import base64
from tests.test_image_generation_and_upload import DummyResponse
import rlbc_daily_to_notion as rlbc


def test_generate_image_on_legion_refiner_model_fallback(monkeypatch):
    calls = []
    fake_bytes = b'ref-fallback-bytes'
    fake_b64 = base64.b64encode(fake_bytes).decode('utf-8')
    data_url = f"data:image/png;base64,{fake_b64}"

    def fake_post(url, json=None, timeout=None):
        calls.append({'url': url, 'json': json})
        if url.endswith('/API/GetNewSession'):
            return DummyResponse({'session_id': 'sess-ref-fallback'}, status_code=200, ok=True)

        if url.endswith('/API/GenerateText2Image'):
            print('FAKE_POST RECEIVED:', json.get('refiner_model'), json.get('refiner_upscale_method'))
            # If we set a refiner_model that matches configured one, simulate server rejecting it
            if json.get('refiner_model') == f"model-{rlbc.LEGION_REFINER_UPSCALE_METHOD}":
                return DummyResponse({}, status_code=400, ok=False, text=f"Invalid model value for param Refiner Model - 'model-{rlbc.LEGION_REFINER_UPSCALE_METHOD}'")
            # Accept when we use pixel-lanczos as an alternate upscaler
            if json.get('refiner_upscale_method') == 'pixel-lanczos':
                return DummyResponse({'images': [data_url]}, status_code=200, ok=True)
            # Otherwise reject
            return DummyResponse({}, status_code=400, ok=False, text="Refiner model not available")

        return DummyResponse({}, status_code=404, ok=False)

    monkeypatch.setattr(rlbc.requests, 'post', fake_post)

    out, url = rlbc.generate_image_on_legion("Ref Fallback Title", "desc")
    assert out == fake_bytes

    # Verify that at least one attempt used the fallback candidate 'pixel-lanczos'
    gen_calls = [c for c in calls if c['url'].endswith('/API/GenerateText2Image')]

    assert any(c['json'].get('refiner_upscale_method') == 'pixel-lanczos' for c in gen_calls)
