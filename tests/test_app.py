from app.app import predict_post


def test_app():
    assert predict_post() == "Hello, world!"

