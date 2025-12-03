from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import time

# model = os.environ.get("MODEL", "llama2-uncensored")
# model = os.environ.get("MODEL", "llama2")
model = os.environ.get("MODEL", "mistral-openorca")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

llm = OllamaLLM(model=model)

# RAG prompt template
rag_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""

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
    query = f"""what are some related questions that can be asked for the parent question '{parent_question}' based on the following related context:\n{related_context}. Only include the related questions in a numbered format"""
    res = llm.invoke(query)
    return res

def extract_related_context(docs):
    related_context = ""
    for document in docs:
        related_context += f"\n> {document.metadata['source']}:\n{document.page_content}\n"
    return related_context

def generate_rag_response(query):
    s = time.time()
    # Get source documents
    docs = retriever.invoke(query)
    # Get answer from RAG chain
    answer = rag_chain.invoke(query)
    e = time.time()
    print(f"took {e-s} seconds to complete RAG based QA")

    s = time.time()
    related_context = extract_related_context(docs)
    related_questions = generate_related_questions(query, related_context)
    e = time.time()
    print(f"took {e-s} seconds to generate related_questions")
    
    return answer, related_questions
