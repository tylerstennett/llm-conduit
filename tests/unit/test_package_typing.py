from __future__ import annotations

from importlib import resources


def test_py_typed_marker_is_packaged() -> None:
    marker = resources.files("conduit").joinpath("py.typed")
    assert marker.is_file()
