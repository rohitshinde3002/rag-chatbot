import os
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


def get_qa_chain(vectorstore):

    # ✅ Groq LLM
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.2
    )

    # ✅ Retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    # ✅ Prompt (improved format)
    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant.

        Answer ONLY using the given context.
        If answer is not in context, say "I don't know".

        Context:
        {context}

        Question:
        {input}

        Answer:
        """
    )

    # ✅ Document chain
    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    # ✅ Retrieval chain
    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    return retrieval_chain