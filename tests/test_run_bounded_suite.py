from scripts.run_bounded_suite import _installed_airllm_title


def test_airllm_method_title_reflects_the_actually_installed_version(monkeypatch):
    """The airllm Method's title used to be the hardcoded literal "AirLLM
    3.1.0", printed as the run's "METHOD: ..." log line and written into
    every result JSON's title field regardless of which airllm was actually
    running -- so upgrading the installed package (as this project did to
    airllm 3.2.0) silently mislabeled every subsequent result. The title is
    now computed from the installed package at call time."""
    def fake_version(name):
        assert name == "airllm"
        return "9.9.9"

    monkeypatch.setattr("importlib.metadata.version", fake_version)
    assert _installed_airllm_title() == "AirLLM 9.9.9"


def test_airllm_method_title_degrades_gracefully_when_not_installed(monkeypatch):
    def raising_version(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("importlib.metadata.version", raising_version)
    title = _installed_airllm_title()
    assert "unknown" in title.lower()
