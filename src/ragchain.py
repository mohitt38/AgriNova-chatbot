from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate


def build_rag_chain():
    template = """
    You are AgriNova, an intelligent agricultural assistant for farmers.
    An expert agricultural assistant. Use the context below to answer
    questions about crops, soil, irrigation, fertilizers, and pest control.

    If the answer is not in the context, say:
    "Answer is not available in the context."

    Instructions:
    - Read the provided context carefully before answering.
    - Solve the farmer's query in a way that is easy to understand and helpful.
    - Detect the language of the user query and respond in the same language.
    - Use simple, everyday language; avoid technical jargon.
    - Keep your answer approximately 200–300 words.
    - Only provide information supported by the context.
    - Be practical, solution-oriented, and farmer-friendly.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        streaming=True
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    # Modern RAG execution (NO LLMChain)
    def rag_chain(context: str, question: str):
        formatted_prompt = prompt.format(
            context=context,
            question=question
        )
        return llm.invoke(formatted_prompt)

    return rag_chain
