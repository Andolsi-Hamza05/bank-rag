from typing import List
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from src.preprocessing import TextProcessor


class VectorStoreIngestor:
    def __init__(self, embedding_model, model_name, persist_directory):
        self.model_name = model_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model

    def ingest(self, documents: List[Document]):
        preprocessor = TextProcessor()
        embedding_function = SentenceTransformerEmbeddings(
                                model_name=self.embedding_model)
        preprocessed_documents = preprocessor.preprocess_documents(documents)
        Chroma.from_documents(
            documents=preprocessed_documents,
            embedding=embedding_function,
            persist_directory=self.persist_directory
        )
        print(f"""Ingested {len(preprocessed_documents)}
             documents into {self.persist_directory}""")

    def load_from_disk(self):
        embedding_function = SentenceTransformerEmbeddings(
                                model_name=self.embedding_model)
        db = Chroma(persist_directory=self.persist_directory,
                    embedding_function=embedding_function)
        return db
