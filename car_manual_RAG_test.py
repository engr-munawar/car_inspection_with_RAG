# test_swift_rag.py
import os
import json
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ============ CONFIGURATION ============
CAR_MANUAL_DB = "./chroma_db_car"
EMBEDDING_MODEL = "bge-m3"
GROQ_MODEL = "openai/gpt-oss-20b"
# =======================================

def test_swift_manual():
    """Test the Swift manual RAG with specific queries"""
    
    print("=== Suzuki Swift Manual RAG Test ===\n")
    
    # 1. Load vector store
    print("Loading vector store...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=CAR_MANUAL_DB,
        embedding_function=embeddings
    )
    doc_count = len(vectorstore.get()['ids'])
    print(f"✅ Loaded {doc_count} chunks from Suzuki Swift manual")
    
    # 2. Initialize Groq
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    # 3. Create RAG chain
    prompt = ChatPromptTemplate.from_template("""
You are an expert Suzuki mechanic with full access to the Suzuki Swift repair manual.
Answer the question based on the context from the manual.

If the answer is not in the manual, say:
"I cannot find this information in the Suzuki Swift manual. Please consult a certified mechanic."

Context: {context}
Question: {input}

Answer:
""")
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    # 4. Test queries
    test_queries = [
        "What is the recommended engine oil for Suzuki Swift?",
        "What is the torque specification for cylinder head bolts?",
        "How do I replace the front bumper?",
        "What is the tire pressure for Suzuki Swift?",
        "How do I check the clutch free play?",
        "What is the brake disc thickness specification?",
        "How do I replace the headlight bulb?",
        "What is the coolant capacity?"
    ]
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] Query: {query}")
        print("-" * 50)
        
        start_time = time.time()
        response = rag_chain.invoke({"input": query})
        elapsed = time.time() - start_time
        
        print(f"✅ Answer: {response['answer']}")
        print(f"📚 Sources: {len(response['context'])} chunks retrieved")
        print(f"⏱️  Response time: {elapsed:.2f} seconds")
        
        results.append({
            "query": query,
            "answer": response["answer"],
            "source_count": len(response["context"]),
            "response_time": elapsed
        })
    
    # 5. Save results
    with open("swift_manual_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*50)
    print("✅ Test complete! Results saved to swift_manual_test_results.json")
    
    # Summary
    avg_time = sum(r["response_time"] for r in results) / len(results)
    print(f"📊 Average response time: {avg_time:.2f} seconds")

if __name__ == "__main__":
    test_swift_manual()