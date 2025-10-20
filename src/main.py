from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import openai

load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

openai.api_type = "azure"
openai.api_base = AZURE_ENDPOINT
openai.api_version = "2023-07-01-preview"
openai.api_key = AZURE_API_KEY

app = FastAPI(title="Nerd AI Tutor Backend")

# Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5173"] if using Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageRequest(BaseModel):
    message: str

@app.post("/ask")
async def ask_question(request: MessageRequest):
    response = openai.chat_completions.create(
        engine=AZURE_DEPLOYMENT_NAME,
        messages=[{"role": "user", "content": request.message}],
        temperature=0.7,
        max_tokens=500,
    )
    answer = response.choices[0].message["content"]
    return {"answer": answer}


@app.post("/upload-homework")
async def upload_homework(file: UploadFile = File(...), message: str = Form(...)):
    # Save file locally (or send to cloud storage like Azure Blob)
    file_location = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Send message to AI (optional: include homework info)
    response = openai.chat_completions.create(
        engine=AZURE_DEPLOYMENT_NAME,
        messages=[{"role": "user", "content": message}],
        temperature=0.7,
        max_tokens=500,
    )
    answer = response.choices[0].message["content"]

    return {"answer": answer, "file_path": file_location}
