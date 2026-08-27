"""Focused browser contract for the bundled UI.

The test skips when Playwright's optional Chromium binary is not installed;
the Python dependency alone does not download browsers during package install.
"""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

playwright = pytest.importorskip("playwright.sync_api")


@pytest.fixture()
def live_server(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = os.environ.copy()
    env["AFTERIMAGE_STATE_DB"] = str(tmp_path / "state.sqlite3")
    env["AFTERIMAGE_STORE_ROOT"] = str(tmp_path / "stores")
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "afterimage.server.app:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "error"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url + "/", timeout=0.25) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(0.1)
        else:
            pytest.fail("UI test server did not start")
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def _json(route, payload):
    route.fulfill(status=200, content_type="application/json", body=json.dumps(payload))


def test_desktop_and_mobile_product_flow(live_server):
    with playwright.sync_playwright() as runtime:
        try:
            browser = runtime.chromium.launch(headless=True)
        except playwright.Error as exc:
            pytest.skip("Playwright Chromium is not installed: %s" % exc)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page_errors = []
        page.on("pageerror", lambda error: page_errors.append(str(error)))

        def mock_api(route):
            path = route.request.url.split("?", 1)[0]
            if path.endswith("/health"):
                return _json(route, {"status": "ok", "model_loaded": False})
            if path.endswith("/api/version"):
                return _json(route, {"version": "0.3.0"})
            if path.endswith("/api/hardware"):
                return _json(route, {
                    "gpu": {"name": "Test GPU", "vendor": "nvidia"},
                    "vram_total_gb": 8.0, "vram_free_gb": 6.5,
                    "memory": {"total_gib": 32.0, "available_gib": 21.0},
                    "disk": {"total_gib": 1000.0, "free_gib": 600.0, "error": None},
                })
            if path.endswith("/api/capability"):
                return _json(route, {"measured_reference": {
                    "model": "Qwen/Qwen3-14B", "params_b": 14,
                    "bf16_gb_per_b_params": 2.1097,
                    "compressed_gb_per_b_params": 1.452,
                    "fast_s_per_token_per_b": 0.6536,
                }})
            if path.endswith("/api/models"):
                return _json(route, {"models": [{
                    "model_id": "Qwen/Qwen3-VL-8B-Instruct", "state": "ready",
                    "comp_gb": 13.2, "updated_at": 1,
                    "metadata": {"compatibility": {
                        "execution": "experimental", "modality": "vision-text",
                        "mixture_of_experts": False,
                    }},
                    }]})
            if path.endswith("/api/models/discover"):
                return _json(route, {"models": [{
                    "model_id": "qwen3:8b", "source": "ollama",
                    "source_label": "Ollama", "format": "Q4_K_M",
                    "size_bytes": 5_200_000_000, "can_prepare": False,
                    "message": "This Ollama model is already on disk.",
                    "external_url": "http://127.0.0.1:8000/ui",
                }], "sources": {"huggingface_cache": 0, "ollama": 1}})
            if path.endswith("/api/jobs"):
                return _json(route, {"jobs": []})
            if path.endswith("/api/experiment-runs"):
                return _json(route, {"runs": []})
            if path.endswith("/api/runtime-profiles"):
                return _json(route, {"profiles": []})
            if path.endswith("/api/experiments"):
                return _json(route, {"hypotheses": [{
                    "id": "h1-test", "title": "Residency optimization",
                    "statement": "Does measured placement reduce runtime?",
                    "candidate_profile": "candidate", "control_profile": "control",
                    "runner": "generation", "primary_metric": "tokens_per_second",
                    "minimum_effect": 0.02, "minimum_repeats": 3,
                    "minimum_new_tokens": 8, "required_inputs": [],
                    "measured": {"verdict": "positive_screen", "effect_pct": 1.61,
                                 "plain_language": "A positive mechanism screen.",
                                 "detail": "Measured separately from this local run."},
                }]})
            if path.endswith("/api/catalog/models"):
                return _json(route, {"models": [{
                    "model_id": "Qwen/Qwen3-70B", "params_b": 70,
                    "estimated_source_gb": 147.7, "downloads": 12000,
                    "execution": "download-only",
                    "execution_reason": "The checkpoint can still be downloaded.",
                    "modality": "text", "mixture_of_experts": False,
                    "availability": "remote", "action": "get", "gated": False,
                }], "page": 1, "page_window": [1, 2], "next_cursor": "next",
                    "previous_cursor": None, "exhausted": False})
            route.continue_()

        page.route("**/*", mock_api)
        page.goto(live_server, wait_until="networkidle")
        page.locator("#machine-line-body strong").first.wait_for()
        assert "32.0 GB" in page.locator("#machine-line-body").inner_text()
        assert "8.0 GB" in page.locator("#machine-line-body").inner_text()
        assert "?" not in page.locator("#machine-line-body").inner_text()
        # The mocked /api/models above returns exactly one ready model, so
        # Home should be in its "one model ready" state: the model itself,
        # an Afterimage Fit badge, and a direct path into Chat.
        assert "Qwen/Qwen3-VL-8B-Instruct" in page.locator("#home-status").inner_text()
        assert page.locator("#home-status .badge.fit-afterimage-ready").count() == 1
        page.locator("[data-home-chat]").wait_for()

        page.locator('[data-route="models"]').click()
        page.locator("#catalog-query").fill("Qwen")
        page.locator("#catalog-search").evaluate("form => form.requestSubmit()")
        get_button = page.locator("[data-catalog-get]")
        get_button.wait_for()
        assert get_button.is_enabled()
        assert "70B parameters" in page.locator("#catalog-results").inner_text()
        assert "qwen3:8b" in page.locator("#computer-results").inner_text()
        assert page.locator("#catalog-pages button").all_inner_texts() == ["1", "2"]
        assert page.locator("#catalog-results").bounding_box()["y"] < page.locator("#local-models").bounding_box()["y"]

        page.locator('[data-route="chat"]').click()
        assert page.locator("#chat-model").input_value() == "Qwen/Qwen3-VL-8B-Instruct"
        assert page.locator("#attach-button").is_enabled()
        assert page.locator("#chat-stop").is_hidden()
        assert "v1/chat/completions" in page.locator("#chat-endpoint").inner_text()

        page.locator('[data-route="research"]').click()
        page.locator("#research-form").wait_for()
        assert page.locator("#research-form").is_visible()
        assert page.locator(".research-builder").count() == 1
        assert "A positive mechanism screen" not in page.locator("[data-page=research]").inner_text()
        assert page.locator("#research-repeats").input_value() == "3"
        assert page.locator("#report-count").inner_text() == "0"

        page.set_viewport_size({"width": 390, "height": 844})
        page.locator("#mobile-menu").click()
        page.locator('[data-route="models"]').click()
        assert page.locator("#catalog-results").bounding_box()["width"] < 390
        assert page.evaluate("document.documentElement.scrollWidth") == 390
        assert page.locator(".catalog-card").count() == 1
        assert page_errors == []
        browser.close()
