from app.app import app


def test_app():
    assert app() == "Hello, world!"

