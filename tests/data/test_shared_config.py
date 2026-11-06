"""Shared configuration merging must preserve caller-owned data."""

from src.config.yaml import deep_merge


def test_nested_merge_replaces_lists_and_nulls_without_mutating_inputs():
    base = {"limits": {"bytes": 100, "timeout": 30}, "items": [1, 2], "optional": {"a": 1}}
    override = {"limits": {"bytes": 200}, "items": [3], "optional": None, "new": {"values": [4]}}
    merged = deep_merge(base, override)
    assert merged == {"limits": {"bytes": 200, "timeout": 30}, "items": [3], "optional": None, "new": {"values": [4]}}
    merged["limits"]["timeout"] = 90
    merged["items"].append(5)
    merged["new"]["values"].append(6)
    assert base["limits"]["timeout"] == 30
    assert base["items"] == [1, 2]
    assert override["items"] == [3]
    assert override["new"]["values"] == [4]
