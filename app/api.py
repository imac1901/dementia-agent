from __future__ import annotations

import os
from pathlib import Path
from typing import List, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

LOG_PATH = Path("logs/object_detections.txt")
DEFAULT_MODEL = "gpt-5-nano"
MAX_LOG_CHARACTERS = 8000

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(
    title="Home Object Assistant API",
    description=(
        "Submit chat-style prompts to OpenAI's GPT-5 Nano model. "
        "The service automatically prepends the latest object detection logs "
        "as a system message."
    ),
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(
        ...,
        description="The conversational role for the message.",
        examples=["user"],
    )
    content: str = Field(
        ...,
        description="Plain-text message content.",
        min_length=1,
        examples=["Where did I last see the screwdriver?"],
    )


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(
        ...,
        min_items=1,
        description=(
            "Chronological list of chat messages between the user and assistant. "
            "The API automatically injects a system message populated from the "
            "object detection log before these messages."
        ),
        example={
            "messages": [
                {"role": "user", "content": "Summarize recent detections."},
                {
                    "role": "assistant",
                    "content": "Earlier summary from the assistant (optional).",
                },
                {
                    "role": "user",
                    "content": "Focus on kitchen-related items in the latest logs.",
                },
            ]
        },
    )
    model: Optional[str] = Field(
        DEFAULT_MODEL,
        description=(
            "Override the default OpenAI model. "
            "Defaults to GPT-5 Nano."
        ),
        examples=[DEFAULT_MODEL],
    )


def _load_log_context(max_chars: int = MAX_LOG_CHARACTERS) -> str:
    if LOG_PATH.exists():
        log_content = LOG_PATH.read_text(encoding="utf-8", errors="ignore")
        truncated = len(log_content) > max_chars
        if truncated:
            log_content = log_content[-max_chars:]
            prefix = "(Log truncated to last "
            log_header = f"{prefix}{max_chars} characters)\n"
        else:
            log_header = ""
        return f"{log_header}{log_content.strip()}"

    return "No object detection logs are available yet."


@app.post(
    "/query",
    summary="Send a chat prompt to GPT-5 Nano with detection-log context.",
)
async def query(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="Provide at least one message.")

    system_context = (
        "You are an assistant helping users keep track of household objects. "
        "You help the user keep track of objects in their home and provide helpful information about them."
        "You format your answers in a friendly, human readable format, referencing points in time humanly understandable terms."
        "Base your answers on the following object detection log:\n"
        f"{_load_log_context()}"
    )

    input_messages = [{"role": "system", "content": system_context}]
    input_messages.extend(
        {"role": message.role, "content": message.content} for message in request.messages
    )

    model_name = request.model or DEFAULT_MODEL

    try:
        response = client.responses.create(model=model_name, input=input_messages)
    except Exception as exc:  # pragma: no cover - external service errors
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    response_text = getattr(response, "output_text", None)
    if not response_text and getattr(response, "output", None):
        # Fallback: concatenate text segments if output_text is unavailable.
        response_text = "".join(
            part.text for part in response.output if getattr(part, "text", None)
        )

    usage_payload = None
    if getattr(response, "usage", None):
        try:
            usage_payload = response.usage.model_dump()
        except Exception:  # pragma: no cover
            usage_payload = None

    return {
        "model": model_name,
        "response": response_text,
        "usage": usage_payload,
    }


