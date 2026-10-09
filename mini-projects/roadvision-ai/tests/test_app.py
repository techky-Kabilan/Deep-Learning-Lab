from streamlit.testing.v1 import AppTest

def test_app_without_weights(monkeypatch, tmp_path):
    monkeypatch.setenv("ROADVISION_WEIGHTS", str(tmp_path / "missing.pt"))
    app = AppTest.from_file("app.py").run(timeout=20)
    assert not app.exception
    assert app.title[0].value == "RoadVision AI"
    assert "Training pending" in app.info[0].value
