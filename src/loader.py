import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(pdf_paths):

    documents = []

    for pdf in pdf_paths:

        if not os.path.exists(pdf):
            continue

        if os.path.getsize(pdf) == 0:
            raise ValueError(f"Empty PDF file: {pdf}")

        loader = PyPDFLoader(pdf)
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    return chunks