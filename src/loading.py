import os
import glob
from typing import List
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.documents.base import Document


class DocumentLoader:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    def get_all_pdfs(self) -> List[str]:
        """
        Retrieve all PDF files from the specified folder path.
        """
        filenames = glob.glob(os.path.join(self.folder_path, '*.pdf'))
        return filenames

    def load_documents(self, filenames: List[str]) -> List[Document]:
        """
        Load text content from PDF documents using PyPDFLoader.
        """
        documents = []
        for filename in filenames:
            loader = PyPDFLoader(filename)
            docs = loader.load()
            for doc in docs:
                documents.append(doc)
        return documents
