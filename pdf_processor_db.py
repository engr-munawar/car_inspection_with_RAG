# pdf_processor.py
import os
from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ CONFIGURATION ============
PDF_PATH = "docs/swift_car_manual.pdf"  # Your downloaded PDF
PERSISTENT_DIRECTORY = "./chroma_db_car"  # Separate from your previous DB
EMBEDDING_MODEL = "bge-m3"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
# =======================================

def process_car_manual():
    """Load, chunk, and embed a car manual PDF"""
    
    # 1. Check if PDF exists
    if not os.path.exists(PDF_PATH):
        logger.error(f"❌ PDF not found at {PDF_PATH}")
        return
    
    print(f"📄 Processing: {PDF_PATH}")
    
    # 2. Load PDF (try multiple loaders for compatibility)
    try:
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        logger.info(f"✅ Loaded {len(documents)} pages using PyPDFLoader")
    except Exception as e:
        logger.warning(f"PyPDFLoader failed: {e}, trying PDFPlumber...")
        loader = PDFPlumberLoader(PDF_PATH)
        documents = loader.load()
        logger.info(f"✅ Loaded {len(documents)} pages using PDFPlumber")
    
    # 3. Split into chunks (larger chunks for technical content)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"📊 Created {len(chunks)} chunks")
    
    # 4. Initialize embeddings with bge-m3
    print(f"🧠 Generating embeddings with {EMBEDDING_MODEL}...")
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        show_progress=True
    )
    
    # 5. Create vector store
    print("💾 Creating vector store...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSISTENT_DIRECTORY
    )
    
    doc_count = len(vectorstore.get()['ids'])
    print(f"✅ Vector store created with {doc_count} documents")
    print(f"📁 Saved to: {PERSISTENT_DIRECTORY}")
    
    return vectorstore

#if __name__ == "__main__":
    #process_car_manual() # use in case db is to be created