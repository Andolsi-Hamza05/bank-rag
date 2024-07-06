import string
import re
from typing import List
from langchain_core.documents.base import Document
from langchain_text_splitters import CharacterTextSplitter


class TextProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess a single text document.
        """
        # Lowercase the text
        text = text.lower()

        # Remove bullet points and similar characters
        bullet_points = ['●', '▪', '•', '◦', '‣', '∙']
        for bullet in bullet_points:
            text = text.replace(bullet, '')

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove newline characters and redundant spaces
        text = text.replace('\n', ' ').replace('\r', '')

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove extra white spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def chunk_text(self, documents: List[Document]) -> List[List[Document]]:
        """
        Chunk text content of documents into smaller parts using
        CharacterTextSplitter.
        """
        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return text_splitter.split_documents(documents)

    def preprocess_documents(self,
                             documents: List[Document]) -> List[Document]:
        """
        Preprocess a list of documents by applying text preprocessing
        to each document.
        """
        processed_docs = []
        for doc in documents:
            cleaned_text = self.preprocess_text(doc.page_content)
            processed_docs.append(Document(page_content=cleaned_text,
                                           metadata=doc.metadata))
        return processed_docs
