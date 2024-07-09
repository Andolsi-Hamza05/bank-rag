from src.loading import DocumentLoader
from src.vector_database import VectorStoreIngestor
from src.generating import main


if __name__ == "__main__":
    persist_directory = "chroma"
    PATH = "data/"
    model_name = "HuggingFaceH4/zephyr-7b-beta"
    embedding_model = "all-MiniLM-L6-v2"
    query = "what credit cards do you offer?"
    dl = DocumentLoader(PATH)
    vsi = VectorStoreIngestor(embedding_model, model_name, persist_directory)
    filenames = dl.get_all_pdfs()
    documents = dl.load_documents(filenames)
    print(f"got all documents: {len(documents)}")
    vsi.ingest(documents)
    response = main(query)
    print(response)
