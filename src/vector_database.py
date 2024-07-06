from typing import List
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from src.preprocessing import TextProcessor


class VectorStoreIngestor:
    def __init__(self, model_name, persist_directory):
        self.model_name = model_name
        self.persist_directory = persist_directory

    def ingest(self, documents: List[Document]):
        preprocessor = TextProcessor()
        preprocessed_documents = preprocessor.preprocess_documents(documents)
        Chroma.from_documents(
            documents=preprocessed_documents,
            embedding=self.embedding_function,
            persist_directory=self.persist_directory
        )
        print(f"""Ingested {len(preprocessed_documents)}
             documents into {self.persist_directory}""")

    def load_from_disk(self):
        embedding_function = SentenceTransformerEmbeddings(self.model_name)
        db = Chroma(persist_directory=self.persist_directory,
                    embedding_function=embedding_function)
        return db
