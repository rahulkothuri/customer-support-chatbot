from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# model = os.environ.get("MODEL", "llama2-uncensored")
# model = os.environ.get("MODEL", "llama2")
model = os.environ.get("MODEL", "mistral-openorca")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

# Cache embeddings model
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, model_kwargs={'device': 'cpu'})
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

# Use num_predict to limit response length for faster generation
llm = OllamaLLM(model=model, num_predict=200, temperature=0.3)
llm_related = OllamaLLM(model=model, num_predict=100, temperature=0.5)  # Shorter for related questions

# RAG prompt template - optimized for X/Twitter support
rag_template = """You are X (Twitter) Support Assistant. Answer the user's question using ONLY the context provided.

Rules:
- Be helpful, friendly, and concise (2-4 sentences max)
- Use bullet points for steps
- If the context doesn't contain the answer, say "I don't have specific information about that. Please visit help.x.com for more details."
- Never make up information

Context:
{context}

User Question: {question}

Response:"""

rag_prompt = PromptTemplate.from_template(rag_template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create RAG chain using LCEL
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)


def generate_related_questions(parent_question, related_context):
    query = f"""Based on this X/Twitter support topic, suggest exactly 3 follow-up questions the user might ask. Keep each question under 10 words.

Topic: {parent_question}

1.
2.
3."""
    res = llm_related.invoke(query)
    return res

def extract_related_context(docs):
    related_context = ""
    for document in docs:
        related_context += f"\n> {document.metadata['source']}:\n{document.page_content}\n"
    return related_context

def generate_rag_response(query, include_related=True):
    s = time.time()
    # Get source documents first (needed for both tasks)
    docs = retriever.invoke(query)
    related_context = extract_related_context(docs)
    
    # Run RAG answer and related questions in parallel
    if include_related:
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            future_answer = executor.submit(rag_chain.invoke, query)
            future_related = executor.submit(generate_related_questions, query, related_context)
            
            # Get results
            answer = future_answer.result()
            related_questions = future_related.result()
    else:
        answer = rag_chain.invoke(query)
        related_questions = ""
    
    e = time.time()
    print(f"took {e-s} seconds total (parallel execution)")
    
    return answer, related_questions
