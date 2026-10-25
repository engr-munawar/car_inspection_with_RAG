# Import necessary libraries
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import os
import json
from dotenv import load_dotenv

# Existing imports
from utils.car_inspection import CarInspection
from utils.prepare_assessment_responce import prepare_assessment_response
from utils.json_storage import load_assessments, save_assessments
from utils.cleanup_temp_folder import cleanup_temp_folder

# RAG imports
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

app = FastAPI(title="Vehicle Inspection + Suzuki Swift RAG API", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create temp directory
os.makedirs("temp_images", exist_ok=True)

# assessment.json file storage
assessments_db = load_assessments()

# ============ RAG CONFIGURATION ============
CAR_MANUAL_DB = "./chroma_db_car"
EMBEDDING_MODEL = "bge-m3"
GROQ_MODEL = "openai/gpt-oss-20b"

# Global RAG variables
rag_chain = None
rag_available = False

# ============ RAG INITIALIZATION ============
@app.on_event("startup")
async def startup_event():
    global rag_chain, rag_available
    
    print("🚀 Starting Vehicle Inspection + Suzuki Swift RAG API...")
    
    # Check if RAG vector store exists
    if os.path.exists(CAR_MANUAL_DB) and os.listdir(CAR_MANUAL_DB):
        try:
            print(f"📚 Loading Suzuki Swift manual from {CAR_MANUAL_DB}...")
            
            # Initialize embeddings
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
            
            # Load vector store
            vectorstore = Chroma(
                persist_directory=CAR_MANUAL_DB,
                embedding_function=embeddings
            )
            doc_count = len(vectorstore.get()['ids'])
            print(f"✅ Loaded {doc_count} chunks from Suzuki Swift manual")
            
            # Initialize retriever
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            
            # Initialize Groq LLM
            print(f"🤖 Initializing Groq LLM with model: {GROQ_MODEL}...")
            llm = ChatGroq(
                model=GROQ_MODEL,
                temperature=0,
                api_key=os.getenv("GROQ_API_KEY")
            )
            
            # Create RAG chain
            prompt = ChatPromptTemplate.from_template("""
You are an expert Suzuki mechanic with full access to the Suzuki Swift repair manual.
Answer the question based on the context from the manual.

If the answer is not in the manual then try again one more time to find the answer by rephrasing the question after removing "_", "left" and "right" from part name for example if missing or damaged part is front_bumper then you can rephrase the question"front bumper repair procedure?" and search again in the manual. Similarly for other parts like headlight_left or headlight_right, you can rephrase the question as "headlight repair procedure?" and search again in the manual. If still not found, then say "I cannot find this information in the Suzuki Swift manual. Please consult a certified mechanic".

Context: {context}
Question: {input}

Answer:
""")
            
            document_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, document_chain)
            rag_available = True
            print("✅ RAG chain ready!")
            
        except Exception as e:
            print(f"⚠️ Could not load car manual RAG: {e}")
            rag_available = False
    else:
        print(f"⚠️ Car manual vector store not found at {CAR_MANUAL_DB}")
        print("   Run pdf_processor.py to create the vector store first.")
        rag_available = False

# ============ ENDPOINTS ============

@app.get("/")
async def root():
    return {
        "message": "Vehicle Inspection + Suzuki Swift RAG API is running",
        "status": "healthy",
        "rag_available": rag_available
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "rag_available": rag_available,
        "rag_model": GROQ_MODEL if rag_available else None,
        "manual_chunks": "9880 chunks from Suzuki Swift manual" if rag_available else None
    }

# ============ RAG STATUS ENDPOINT ============
@app.get("/rag-status")
async def rag_status():
    """Check RAG system status and get information about the Suzuki Swift manual"""
    
    if not rag_available:
        return {
            "status": "unavailable",
            "message": "Suzuki Swift manual RAG is not loaded.",
            "suggestion": "Run pdf_processor_db.py to create the vector store.",
            "vector_store_path": CAR_MANUAL_DB
        }
    
    return {
        "status": "available",
        "manual_source": "Suzuki Swift (2,513 pages, 9,880 chunks)",
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": GROQ_MODEL,
        "response_time_avg": "~3.67 seconds"
    }

# ============ RAG QUERY ENDPOINT ============
@app.post("/ask-manual")
async def ask_manual(
    query: str = Form(..., description="Your question about the Suzuki Swift manual")
):
    """
    Ask a question about the Suzuki Swift repair manual.
    Returns repair procedures, specifications, and maintenance information.
    """
    
    if not rag_available:
        raise HTTPException(
            status_code=503,
            detail="Suzuki Swift manual RAG is not available. Please check /rag-status."
        )
    
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        response = rag_chain.invoke({"input": query})
        
        return {
            "query": query,
            "answer": response["answer"],
            "source_count": len(response["context"]),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============ RAG TEST QUERIES ENDPOINT ============
@app.get("/test-rag")
async def test_rag():
    """Run a set of test queries to verify RAG is working correctly"""
    
    if not rag_available:
        raise HTTPException(status_code=503, detail="RAG is not available")
    
    test_queries = [
        "What is the torque specification for cylinder head bolts?",
        "How do I replace the front bumper?",
        "What is the brake disc thickness specification?",
        "How do I check the clutch free play?",
        "What is the coolant capacity?"
    ]
    
    results = []
    for query in test_queries:
        response = rag_chain.invoke({"input": query})
        results.append({
            "query": query,
            "answer": response["answer"][:300] + "..." if len(response["answer"]) > 300 else response["answer"],
            "source_count": len(response["context"])
        })
    
    return {
        "test_results": results,
        "timestamp": datetime.now().isoformat()
    }

# ============ INSPECTION WITH RAG INTEGRATION ============
@app.post("/inspect-vehicle")
async def inspect_vehicle(
    # Required images
    front: UploadFile = File(..., description="Front view image (REQUIRED)"),
    back: UploadFile = File(..., description="Back view image (REQUIRED)"),
    left: UploadFile = File(..., description="Left view image (REQUIRED)"),
    right: UploadFile = File(..., description="Right view image (REQUIRED)"),
    # Optional images
    angle_1: Union[UploadFile, None, str] = File(None),
    angle_2: Union[UploadFile, None, str] = File(None),
    angle_3: Union[UploadFile, None, str] = File(None),
    # NEW: RAG integration option
    get_repair_info: bool = Form(False, description="Get repair procedures from Suzuki Swift manual")
):
    """
    Inspect vehicle using YOLO detection.
    Optionally get repair procedures from the Suzuki Swift manual.
    """
    
    # Cleanup temp folder
    cleanup_temp_folder()
    
    REQUIRED = {
        "front": front,
        "back": back,
        "left": left,
        "right": right
    }

    OPTIONAL = {
        "angle_1": angle_1,
        "angle_2": angle_2,
        "angle_3": angle_3
    }

    inspection = CarInspection()

    # Save + process required images
    for view, file in REQUIRED.items():
        if file is None:
            raise HTTPException(400, f"Missing required image: {view}")

        save_path = f"temp_images/{inspection.car_id}_{view}.jpg"
        with open(save_path, "wb") as f:
            f.write(await file.read())

        assessment = inspection.process_image(save_path, view)
        inspection.assessment["images"][view] = assessment
        inspection._update_summary(None)

    # Process optional views
    for view, file in OPTIONAL.items():
        if file is None:
            continue

        if isinstance(file, str) and file.strip().lower() in ("", "none", "false"):
            continue

        try:
            save_path = f"temp_images/{inspection.car_id}_{view}.jpg"
            if hasattr(file, "read"):
                content = await file.read()
                with open(save_path, "wb") as f:
                    f.write(content)
            else:
                continue

            assessment = inspection.process_image(save_path, view)
            inspection.assessment['images'][view] = assessment
            inspection._update_summary(None)

        except Exception as e:
            print(f"⚠️ Error processing optional view {view}: {e}")
            inspection.assessment[view] = {
                "view_type": view,
                "error": str(e)
            }

    # Combine license plates
    inspection.combine_license_plates()

    # Convert result for JSON
    final = prepare_assessment_response(inspection.assessment)
    
    # ============ RAG INTEGRATION ============
    if get_repair_info and rag_available:
        try:
            # Get damaged parts from assessment
            damaged_parts = final.get("summary", {}).get("damaged_parts", [])
            missing_parts = final.get("summary", {}).get("missing_parts", [])
            
            # Combine parts for repair query
            all_parts = list(set(damaged_parts + missing_parts))
            
            if all_parts:
                # Build repair queries for each damaged part
                repair_info = {}
                
                for part in all_parts[:3]:  # Limit to first 3 parts to avoid too many queries
                    query = f"{part} repair procedure?"
                    
                    try:
                        response = rag_chain.invoke({"input": query})
                        repair_info[part] = {
                            "procedure": response["answer"],
                            "source_count": len(response["context"])
                        }
                    except Exception as e:
                        repair_info[part] = {
                            "error": str(e)
                        }
                
                # Add repair info to the response
                final["repair_information"] = repair_info
                final["rag_enabled"] = True
            else:
                final["repair_information"] = "No damaged or missing parts detected. Vehicle appears to be in good condition."
                final["rag_enabled"] = True
                
        except Exception as e:
            print(f"⚠️ RAG integration error: {e}")
            final["repair_information"] = "Could not retrieve repair information."
            final["rag_enabled"] = False
    elif get_repair_info and not rag_available:
        final["repair_information"] = "RAG system is not available. Please check /rag-status."
        final["rag_enabled"] = False
    else:
        final["repair_information"] = "RAG not requested. Set get_repair_info=true to get repair procedures."
        final["rag_enabled"] = False

    # Save assessment
    assessments_db[inspection.car_id] = final
    save_assessments(assessments_db)

    return {
        "message": "Inspection completed successfully",
        "car_id": inspection.car_id,
        "uploaded_images": {
            "required": list(REQUIRED.keys()),
            "optional": [v for v, f in OPTIONAL.items() if f is not None]
        },
        "assessment": final
    }

# ============ EXISTING ENDPOINTS (Unchanged) ============

@app.get("/assessment/{car_id}")
async def get_assessment(car_id: str):
    """Retrieve assessment by car ID"""
    if car_id not in assessments_db:
        raise HTTPException(404, "Assessment not found")
    
    return assessments_db[car_id]

@app.get("/assessments")
async def list_assessments():
    """List all assessments"""
    return assessments_db

# ============ RUN SERVER ============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )