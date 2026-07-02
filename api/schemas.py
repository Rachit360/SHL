"""
API Schemas
-----------
Strict Pydantic models matching the SHL evaluator's expected contract.
"""

from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message content cannot be blank")
        return v.strip()


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)

    @model_validator(mode="after")
    def last_message_is_user(self) -> "ChatRequest":
        if self.messages and self.messages[-1].role != "user":
            raise ValueError("Last message must have role 'user'")
        return self


class Recommendation(BaseModel):
    name: str = Field(..., description="Assessment name from SHL catalog")
    url: str  = Field(..., description="Full catalog URL")
    test_type: str = Field(..., description="Primary type code A/B/C/D/E/K/M/P/S",
                           pattern=r"^[ABCDEKMPSabcdekmpqs]$")

    @field_validator("url")
    @classmethod
    def url_must_be_shl(cls, v: str) -> str:
        if not v.startswith("https://www.shl.com"):
            raise ValueError(f"URL must be from www.shl.com, got: {v!r}")
        return v

    @field_validator("test_type")
    @classmethod
    def uppercase_type(cls, v: str) -> str:
        return v.upper()


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Agent conversational response")
    recommendations: list[Recommendation] = Field(
        default_factory=list,
        max_length=10,
    )
    end_of_conversation: bool = Field(default=False)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
