from typing import List
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from src.preprocessing import TextProcessor


class VectorStoreIngestor:
    def __init__(self, model_name, persist_directory):
        self.model_name = model_name
        self.persist_directory = persist_directory
        self.embedding_function = SentenceTransformer(model_name)

    def ingest(self, documents: List[Document]):
        preprocessor = TextProcessor()
        preprocessed_documents = preprocessor.preprocess_documents(documents)
        db = Chroma.from_documents(
            documents=preprocessed_documents,
            embedding=self.embedding_function.encode,
            persist_directory=self.persist_directory
        )
        print(f"""Ingested {len(preprocessed_documents)}
             documents into {self.persist_directory}""")
        return db.as_retriever(search_type="mmr")
