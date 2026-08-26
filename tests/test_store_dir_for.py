"""_store_dir_for() turns a client-supplied model_id into a filesystem path
under the store root. model_id ultimately comes from the web API's model-
acquisition endpoints, so it must be safe against a hostile value.

Two real gaps existed before this file: only "/" was replaced, so a literal
backslash (a path separator on Windows) walked straight out of the store
root; and even with every separator replaced, a model_id that is *itself*
exactly ".." has no separator to replace and still resolves to the parent
directory once joined. The fix is a resolved-path containment check, not
just character substitution, and these tests exercise both the substitution
and (more importantly) the containment guarantee directly.
"""
from __future__ import annotations

import pytest

from afterimage.cli import _store_dir_for


def test_ordinary_org_slash_name_model_id_stays_under_root(tmp_path):
    result = _store_dir_for("Qwen/Qwen3-14B", tmp_path)
    assert result == (tmp_path / "Qwen__Qwen3-14B").resolve()
    assert result.parent == tmp_path.resolve()


@pytest.mark.parametrize("payload", [
    "../../etc/passwd",
    "..\\..\\Windows\\System32",
    "../../../secret",
    "....//....//etc",
    "a/../../../b",
    "a\\..\\..\\..\\b",
])
def test_traversal_payloads_with_a_separator_are_neutralized_by_substitution(
        tmp_path, payload):
    """Any payload containing at least one "/" or "\\" is fully defused by
    the character replacement alone: with every separator gone, whatever is
    left (including a literal ".." substring) is one flat filename with
    nowhere to walk up from. This must return a contained path, not raise --
    a real "org/name"-shaped model_id can legitimately contain "..' as text
    ("some-org/..experimental-model")."""
    root = tmp_path.resolve()
    result = _store_dir_for(payload, tmp_path)
    assert result == root or root in result.parents, (
        f"{payload!r} resolved to {result}, outside {root}")


def test_a_payload_that_cannot_be_neutralized_by_substitution_still_raises(tmp_path):
    """model_id == ".." has no "/" or "\\" to replace at all, so the
    character-substitution half of the fix does nothing here -- only the
    containment check catches it. This is the case that proves the
    containment check is load-bearing, not redundant with the substitution."""
    with pytest.raises(ValueError, match="outside the store root"):
        _store_dir_for("..", tmp_path)


def test_backslash_is_replaced_the_same_as_forward_slash(tmp_path):
    """The exact gap this fix closes: the original code only replaced "/",
    leaving a literal backslash -- a real path separator on Windows --
    completely unhandled. Confirms both characters produce the identical
    contained result, not just that neither one currently raises."""
    root = tmp_path.resolve()
    forward = _store_dir_for("../../Windows/System32", tmp_path)
    backward = _store_dir_for("..\\..\\Windows\\System32", tmp_path)
    assert forward == backward
    assert root in forward.parents


def test_default_store_root_is_used_when_none_given(monkeypatch, tmp_path):
    monkeypatch.setattr("afterimage.cli.DEFAULT_STORE_ROOT", tmp_path)
    result = _store_dir_for("org/model")
    assert result == (tmp_path / "org__model").resolve()
