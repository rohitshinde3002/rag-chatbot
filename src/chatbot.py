import os
from langchain_groq import ChatGroq

from langchain.chains.combine_documents import (
    create_stuff_documents_chain
)

from langchain.chains.retrieval import (
    create_retrieval_chain
)

from langchain_core.prompts import ChatPromptTemplate


def get_qa_chain(vectorstore):

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.2

    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question only from the provided context.

        Context:
        {context}

        Question:
        {input}
        """
    )

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    return retrieval_chain