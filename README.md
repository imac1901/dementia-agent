# Dementia Agent

This project combines a YOLOX object detector with a FastAPI service so caregivers can ask where an object was last seen. The detector watches a webcam, logs what it notices, and the API uses OpenAI responses to turn those logs into helpful answers.

---

## Prerequisites
- Python 3.11 or later (`python --version`)
- A working webcam (for live detections)
- An OpenAI API key (`https://platform.openai.com/account/api-keys`)

Optional GPU acceleration depends on your ONNX Runtime install. See the [ONNX Runtime docs](https://onnxruntime.ai/docs/) for details.

---

## Step 1: Create and Activate a Virtual Environment

A **virtual environment** (often shortened to “venv”) is a private copy of Python that lives inside your project folder. It keeps your project’s packages separate from the rest of your computer so different projects do not break each other.

Create the venv in the project root:

```powershell
python -m venv .venv
```

Activate it (only one is needed before each work session):
- PowerShell: `.venv\Scripts\Activate.ps1`
- Command Prompt: `.venv\Scripts\activate.bat`
- macOS/Linux: `source .venv/bin/activate`

You can read more about venvs in the [Python documentation](https://docs.python.org/3/library/venv.html).

When you are done working, type `deactivate` to leave the virtual environment.

---

## Step 2: Install Dependencies

With the venv active:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

Key tools you are installing:
- FastAPI for the web API ([docs](https://fastapi.tiangolo.com/))
- Uvicorn to run the server ([docs](https://www.uvicorn.org/))
- OpenCV for camera access and drawing boxes ([docs](https://docs.opencv.org/))
- ONNX Runtime to run the YOLOX model ([docs](https://onnxruntime.ai/docs/))
- OpenAI’s Python SDK to call language models ([docs](https://platform.openai.com/docs/api-reference))

The first run downloads the YOLOX ONNX model (~25 MB) from Hugging Face.

---

## Step 3: Configure Environment Variables

Create a file named `.env` in the project root with your OpenAI key:

```
OPENAI_API_KEY=sk-your-key-here
# Optional: change how often detections are saved (seconds)
# YOLOX_LOG_INTERVAL=1.0
```

The application automatically loads this file using `python-dotenv`.

---

## Step 4: Run the Detector + API

Start the FastAPI server with Uvicorn:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

What happens:
1. `app/detector.py` starts a background thread that opens your default webcam.
2. Frames are analyzed with YOLOX; detections are drawn in a window titled “YOLOX Live”.
3. Detection summaries are appended to `app/logs/detections.txt`.
4. `app/main.py` exposes two API endpoints:
   - `POST /chat` takes a prompt and returns a human-friendly answer based on recent detections.
   - `GET /health` returns `{"status": "ok"}` for readiness probes.

You can test the API with the FastAPI interactive docs at `http://127.0.0.1:8000/docs`.

Stop the server with `Ctrl+C`. Close the detector window or press `q` to stop logging.

---

## Working With Detection Logs

Each log entry shows the UTC timestamp and the highest confidence score per detected object. You can inspect or clear the log by editing `app/logs/detections.txt`. Truncated excerpts are passed to the OpenAI model when answering questions.

---

## Troubleshooting

- **Webcam busy**: Make sure no other program is using the camera. Restart the app if the window closes unexpectedly.
- **CUDA/GPU errors**: ONNX Runtime automatically falls back to CPU. Check the [GPU setup guide](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) if you need acceleration.
- **OpenAI auth errors**: Confirm `OPENAI_API_KEY` is set in `.env` and the venv is active before running `uvicorn`.

---

## Next Steps

- Explore the detection history to understand object patterns.
- Customize logging intervals via `YOLOX_LOG_INTERVAL`.
- Deploy behind a reverse proxy for remote access (FastAPI deployment guide [here](https://fastapi.tiangolo.com/deployment/)).

Happy building!

