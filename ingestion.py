from dotenv import load_dotenv
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
from constant import INDEX_NAME, FILE_PATH


def ingestion_docs():
    load_dotenv()
    path = FILE_PATH
    loader = PyPDFDirectoryLoader(path=path, recursive=True)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=30,
        separators=["\n\n", " \n", " ", "", " \n \n", "  "],
        keep_separator=False,
        length_function=len,
    )
    raw_docs = loader.load()
    print(f"Loaded {len(raw_docs)} documents from given path")
    split_docs = text_splitter.split_documents(raw_docs)
    print(f"Splitted {len(split_docs)} documents")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print(f"Going to add {len(split_docs)} to Pinecone")
    PineconeVectorStore.from_documents(split_docs, embeddings, index_name=INDEX_NAME)
    print("****Loading to vectorstore done ***")


if __name__ == "__main__":
    ingestion_docs()
