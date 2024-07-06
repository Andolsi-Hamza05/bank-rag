from langchain.llms import HuggingFaceEndpoint
from langchain.chat_models import ChatHuggingFace
from utils import get_prompt_template
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from src.vector_database import VectorStoreIngestor


def instanciate_hf_endpoint(repo_id="HuggingFaceH4/zephyr-7b-beta",
                            max_length=1600,
                            temperature=0.01):
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        max_length=max_length,
        temperature=temperature,
    )


def instanciate_hf_llm(endpoint):
    return ChatHuggingFace(llm=endpoint)


def main(query: str):
    PATH = "prompts/naive_rag.txt"
    model_name = "HuggingFaceH4/zephyr-7b-beta"
    persist_directory = "chroma"
    endpoint = instanciate_hf_endpoint()
    llm = instanciate_hf_llm(endpoint)
    template = get_prompt_template(PATH)
    prompt = ChatPromptTemplate.from_template(template)
    vsi = VectorStoreIngestor(model_name, persist_directory)
    db = vsi.load_from_disk()
    baseline_retriever = db.as_retriever(search_type="mmr")
    baseline = (
        {"context": baseline_retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    baseline.invoke(query)
