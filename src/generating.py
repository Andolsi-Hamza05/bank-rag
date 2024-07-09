# generating.py

from langchain.llms import HuggingFaceEndpoint
from langchain.chat_models import ChatHuggingFace
from src.utils import get_prompt_template
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from src.vector_database import VectorStoreIngestor


class LangChainApp:
    def __init__(self,
                 prompt_path="prompts/naive_rag.txt",
                 model_name="HuggingFaceH4/zephyr-7b-beta",
                 persist_directory="chroma",
                 embedding_model="all-MiniLM-L6-v2"):
        self.prompt_path = prompt_path
        self.model_name = model_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model

        self.endpoint = self.instanciate_hf_endpoint()
        print("Instantiated successfully endpoint")
        self.llm = self.instanciate_hf_llm(self.endpoint)
        print("Instantiated successfully llm")
        self.template = get_prompt_template(self.prompt_path)
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.vsi = VectorStoreIngestor(self.embedding_model,
                                       self.model_name,
                                       self.persist_directory)
        self.db = self.vsi.load_from_disk()
        print("Loaded successfully data from disk -- chroma")
        self.baseline_retriever = self.db.as_retriever(search_type="mmr")

    def instanciate_hf_endpoint(self, repo_id="HuggingFaceH4/zephyr-7b-beta",
                                max_length=1600,
                                temperature=0.01):
        return HuggingFaceEndpoint(
            repo_id=repo_id,
            max_length=max_length,
            temperature=temperature,
        )

    def instanciate_hf_llm(self, endpoint):
        return ChatHuggingFace(llm=endpoint)

    def main(self, query: str):
        baseline = (
            {"context": self.baseline_retriever,
             "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return baseline.invoke(query)


langchain_app = LangChainApp()


def run_main(query: str):
    return langchain_app.main(query)
