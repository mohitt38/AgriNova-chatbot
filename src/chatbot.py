from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from src.ragchain import build_rag_chain


def ask_crop_expert_streaming(question, docs):
    context = "\n".join([doc.page_content for doc in docs])

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
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
    docs = vectorstore.similarity_search(question, k=k)

    # NON-STREAMING PATH (RETURNS STRING)
    if not stream:
        if docs:
            qa_chain = build_rag_chain()
            result = qa_chain(
                {"input_documents": docs, "question": question},
                return_only_outputs=True
            )
            return result["output_text"]

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3
        )
        return llm.predict(question)

    # STREAMING PATH (RETURNS GENERATOR)
    if docs:
        return ask_crop_expert_streaming(question, docs)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )
    return llm.stream(question)
