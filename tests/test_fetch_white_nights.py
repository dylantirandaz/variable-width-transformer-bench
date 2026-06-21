import importlib.util
from pathlib import Path


def load_fetcher():
    script = Path(__file__).resolve().parents[1] / "scripts" / "fetch_white_nights.py"
    spec = importlib.util.spec_from_file_location("fetch_white_nights", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_white_nights_only_keeps_story_section() -> None:
    fetcher = load_fetcher()
    html = """
    <p>contents</p>
    <p><a name="Pg1">WHITE NIGHTS</a></p>
    <p class="center"><span>FIRST NIGHT</span></p>
    <p>It was a wonderful night,<br />dear reader.</p>
    <p>My God, a whole moment of happiness!</p>
    <p class="p4 center">NOTES FROM UNDERGROUND</p>
    <p>I am a sick man.</p>
    """

    text = fetcher.extract_white_nights(html)

    assert "WHITE NIGHTS" in text
    assert "FIRST NIGHT" in text
    assert "It was a wonderful night, dear reader." in text
    assert "My God, a whole moment of happiness!" in text
    assert "I am a sick man" not in text
