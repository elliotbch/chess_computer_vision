"""The static site runs copies of the repository's Python code: check they are up to date.

    python -m pytest site/tests          (or: python site/tests/test_sync.py)

If a test fails, run python site/sync_site.py.
"""

import filecmp
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.dirname(HERE)
ROOT = os.path.dirname(SITE)
sys.path.insert(0, SITE)
import sync_site  # noqa: E402

HINT = "run python site/sync_site.py"


def same(a, b):
    return os.path.exists(b) and filecmp.cmp(a, b, shallow=False)


def test_pipeline_copies_are_identical_to_src():
    for m in sync_site.MODULES:
        assert same(os.path.join(ROOT, "src", m), os.path.join(SITE, "py", m)), f"site/py/{m} differs from src/{m}: {HINT}"


def test_glue_is_extracted_verbatim_from_demo_app():
    with open(os.path.join(SITE, "py", "demo_glue.py"), encoding="utf-8") as f:
        assert f.read() == sync_site.glue_source(), f"site/py/demo_glue.py differs from demo/app.py: {HINT}"


def test_manifest_lists_every_python_file():
    with open(os.path.join(SITE, "py", "manifest.json")) as f:
        listed = json.load(f)
    assert listed == sorted(f for f in os.listdir(os.path.join(SITE, "py")) if f.endswith(".py"))
    assert "web_demo.py" in listed and "demo_glue.py" in listed


def test_catalog_examples_and_data():
    with open(os.path.join(SITE, "assets", "catalog.json"), encoding="utf-8") as f:
        assert json.load(f) == sync_site.catalog(), f"site/assets/catalog.json is stale: {HINT}"
    cat = sync_site.catalog()
    ex = os.path.join(ROOT, "demo", "static", "examples")
    for e in cat["examples"]:
        assert same(os.path.join(ex, e["file"]), os.path.join(SITE, "assets", "examples", e["file"])), e["file"]
        assert os.path.exists(os.path.join(SITE, e["pixels"])), e["pixels"]
    for g in cat["games"]:
        assert same(os.path.join(ex, "games", g["file"]), os.path.join(SITE, "assets", "examples", "games", g["file"])), g["file"]
    assert same(os.path.join(ROOT, "training", "metrics.json"), os.path.join(SITE, "assets", "metrics.json"))
    for w in sync_site.WEIGHTS:
        src = os.path.join(ROOT, "models", w)
        if os.path.exists(src):          # models/ is not versioned: checked only where it exists
            assert same(src, os.path.join(SITE, "assets", w)), f"site/assets/{w} differs from models/{w}: {HINT}"
        assert os.path.exists(os.path.join(SITE, "assets", w))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
