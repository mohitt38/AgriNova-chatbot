from src.embed_store import load_vectorstore
from src.ragchain import build_rag_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Streaming helper
def ask_crop_expert_streaming(question, docs):
    context = "\n".join([doc.page_content for doc in docs])
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        streaming=True
    )

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="You are an expert agricultural assistant.\nContext:\n{context}\nQuestion:\n{question}\nAnswer:"
    )

    prompt = prompt_template.format(context=context, question=question)

    # Stream from LLM: yields strings
    for chunk in llm.stream(prompt):
        yield chunk  # chunk is already a string
        
# Main function

def ask_crop_expert(question, k=3, stream=False):
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(question, k=k)

    if not docs:
        return "No relevant information found in the knowledge base."

    if stream:
        # Streaming: return generator
        return ask_crop_expert_streaming(question, docs)
    else:
        # Normal: non-streaming
        qa_chain = build_rag_chain()
        response = qa_chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        return response["output_text"]
