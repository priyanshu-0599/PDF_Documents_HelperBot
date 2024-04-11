import os
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models.base import ChatOpenAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_pinecone.vectorstores import PineconeVectorStore
from constant import INDEX_NAME
from typing import Any, List, Tuple


def run_llm(query: str, chat_history: List[Tuple[str, Any]] = []) -> Any:
    load_dotenv()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docs_search = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )
    chat_llm = ChatOpenAI(model="gpt-3.5-turbo", verbose=True, temperature=0)
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat_llm,
        chain_type="stuff",
        retriever=docs_search.as_retriever(),
        return_source_documents=True,
    )
    return qa.invoke({"question": query, "chat_history": chat_history})
