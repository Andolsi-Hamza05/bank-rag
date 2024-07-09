import os
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from pyngrok import ngrok
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio
from src.loading import DocumentLoader
from src.vector_database import VectorStoreIngestor
from src.generating import run_main

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    content: str


persist_directory = "chroma"
PATH = "data/"
model_name = "HuggingFaceH4/zephyr-7b-beta"
embedding_model = "all-MiniLM-L6-v2"

if not os.path.exists(persist_directory):
    dl = DocumentLoader(PATH)
    vsi = VectorStoreIngestor(embedding_model, model_name, persist_directory)
    filenames = dl.get_all_pdfs()
    documents = dl.load_documents(filenames)
    print(f"got all documents: {len(documents)}")
    vsi.ingest(documents)
else:
    print(f"The directory '{persist_directory}' already exists.")


@app.post('/chat/')
async def quick_response(msg: Message):
    response = run_main(msg.content)
    print(response)
    return response

if __name__ == "__main__":
    # Start ngrok tunnel
    ngrok_tunnel = ngrok.connect(8000)
    print('Public URL:', ngrok_tunnel.public_url)
    nest_asyncio.apply()
    # Run the FastAPI app with uvicorn
    uvicorn.run(app, port=8000)
