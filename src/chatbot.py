from src.ragchain import build_rag_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from src.ragchain import build_rag_chain


def ask_crop_expert_streaming(question, docs):
    """
    Streaming response generator (Gemini)
    """
    context = "\n".join([doc.page_content for doc in docs])

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are an expert agricultural assistant.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer:"
        )

    ).format(context=context, question=question)

    for chunk in llm.stream(prompt):
        yield chunk


def ask_crop_expert(question, vectorstore, k=3, stream=False):
    """
    Main chatbot function (SAFE)
    """

    )

    prompt = prompt_template.format(
        context=context,
        question=question
    )

    for chunk in llm.stream(prompt):
        yield chunk  # Gemini already streams strings


# Main chatbot function 
def ask_crop_expert(question, vectorstore, k=3, stream=False):

    docs = vectorstore.similarity_search(question, k=k)

    if not docs:
        if stream:
            yield "No relevant information found in the knowledge base."
            return
        return "No relevant information found in the knowledge base."

    context_text = "\n".join([doc.page_content for doc in docs])

    if stream:
        return ask_crop_expert_streaming(question, docs)

    # Non-streaming RAG
    rag_chain = build_rag_chain()
    response = rag_chain(
        context=context_text,
        question=question
    )

    return response.content

    qa_chain = build_rag_chain()
    response = qa_chain(
        {"input_documents": docs, "question": question},
        return_only_outputs=True
    )

    return response["output_text"]
