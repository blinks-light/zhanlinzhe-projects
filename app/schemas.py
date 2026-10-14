"""Request and response schemas for the email classifier API."""

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """A single email body to classify."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Raw email body, between 1 and 5000 characters after trimming.",
    )

    @field_validator("text", mode="before")
    @classmethod
    def _require_string(cls, value: object) -> object:
        if not isinstance(value, str):
            raise ValueError("text must be a string")
        return value

    @field_validator("text")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("text must not be empty or whitespace only")
        return cleaned


class PredictionResponse(BaseModel):
    """Classification result returned by /predict."""

    is_spam: bool
    category: str
