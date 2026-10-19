"""Tests for the demo web page."""


def test_index_page(client) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "智能邮件分类器V9" in response.text
    assert 'id="predict"' in response.text
