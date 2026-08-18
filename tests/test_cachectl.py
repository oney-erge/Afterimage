import platform

from afterimage.bench.cachectl import drop_caches, free_ram_bytes, is_cache_control_available


def test_cache_control_availability_matches_platform():
    available = is_cache_control_available()
    if platform.system() == "Linux":
        assert available is True
    else:
        assert available is False, (
            "cache-drop control must not claim availability on a platform "
            "where it cannot actually drop the page cache"
        )


def test_drop_caches_is_honest_about_non_linux():
    if platform.system() != "Linux":
        assert drop_caches() is False


def test_free_ram_bytes_returns_a_plausible_value_on_this_platform():
    val = free_ram_bytes()
    assert val is None or val > 0
