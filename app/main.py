import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from openai import OpenAI
from pydantic import BaseModel

from .detector import DEFAULT_INTERVAL_SECONDS, YOLOXDetector

load_dotenv()
app = FastAPI()


class ChatRequest(BaseModel):
    prompt: str
    metadata: Optional[dict] = None


class ChatResponse(BaseModel):
    reply: str
    note: str


detector = YOLOXDetector(
    interval_seconds=float(os.getenv("YOLOX_LOG_INTERVAL", DEFAULT_INTERVAL_SECONDS)),
)
openai_client = OpenAI()
MODEL_INSTRUCTIONS = (
    "You help users identify where objects are in their home based on historical detections. "
    "Use the provided detection log entries as clues. When multiple objects share the same "
    "timestamp, assume they were in the same location at that moment (e.g., oven and cell phone "
    "together likely means the phone was left in the kitchen). Explain your reasoning briefly, "
    "respond in friendly, human language, and avoid referencing technical details like logs or "
    "internal tools—focus on guidance the user can act on."
    "your answers should be short and concise"
    "if you don't know the answer, say so"
    "when you use time, do so in human readable format. .e.g. 5 minutes ago, 1 hour ago, or around 10:00am"
)
DETECTION_LOG_MAX_CHARS = 6000


def load_detection_log_excerpt() -> str:
    log_path = detector.log_path
    if not log_path.exists():
        return "No detections log is currently available."

    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    if len(log_text) <= DETECTION_LOG_MAX_CHARS:
        return log_text

    excerpt = log_text[-DETECTION_LOG_MAX_CHARS :]
    return (
        "...(truncated for brevity)...\n"
        f"{excerpt}"
    )


@app.on_event("startup")
async def startup_event() -> None:
    detector.start()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    detector.stop()


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    detection_context = await run_in_threadpool(load_detection_log_excerpt)

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "Detection log excerpt to consult when answering:\n"
                        f"{detection_context}"
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": request.prompt,
                }
            ],
        },
    ]

    try:
        response = await run_in_threadpool(
            lambda: openai_client.responses.create(
                model="gpt-4.1-nano",
                instructions=MODEL_INSTRUCTIONS,
                input=messages,
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Upstream model error: {exc}") from exc

    reply_text = getattr(response, "output_text", None)
    if not reply_text:
        raise HTTPException(status_code=502, detail="Model returned an empty response.")

    note = (
        f"Response generated with detection context from {detector.log_path.name} "
        f"({min(len(detection_context), DETECTION_LOG_MAX_CHARS)} characters)."
    )

    return ChatResponse(reply=reply_text, note=note)


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)

