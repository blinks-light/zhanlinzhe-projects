"""Boundary tests for the /predict and /health endpoints."""

import pytest

SPAM_SAMPLES = [
    "Congratulations! You won a free iPhone. Click the link and claim your prize now!",
    "URGENT: verify your bank account details to receive your lottery winnings.",
    "Buy cheap replica watches at 90% discount, limited offer today.",
]

NORMAL_SAMPLES = [
    ("Can you review the Q3 budget spreadsheet before Friday's meeting?", "work"),
    ("Please share the updated project timeline with the client today.", "work"),
    ("Let's grab coffee on Saturday and catch up, it has been ages.", "personal"),
    ("Thanks for dinner last night, we should do it again soon.", "personal"),
    ("Your package was delivered to the reception desk this afternoon.", "notification"),
    ("Reminder: your dentist appointment is on Monday at 9 AM.", "notification"),
]

INVALID_PAYLOADS = [
    {"text": ""},
    {"text": "   "},
    {"text": "a" * 5001},
    {"text": 12345},
    {"text": ["not", "a", "string"]},
    {"text": None},
    {},
]


@pytest.mark.parametrize("text", SPAM_SAMPLES)
def test_spam_prediction(client, text: str) -> None:
    response = client.post("/predict", json={"text": text})

    assert response.status_code == 200
    body = response.json()
    assert body == {"is_spam": True, "category": "spam"}


@pytest.mark.parametrize(("text", "expected_category"), NORMAL_SAMPLES)
def test_normal_subcategory(client, text: str, expected_category: str) -> None:
    response = client.post("/predict", json={"text": text})

    assert response.status_code == 200
    body = response.json()
    assert body["is_spam"] is False
    assert body["category"] == expected_category


@pytest.mark.parametrize("payload", INVALID_PAYLOADS)
def test_invalid_input(client, payload: dict) -> None:
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "detail" in response.json()


def test_health(client) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "models_loaded": True}
