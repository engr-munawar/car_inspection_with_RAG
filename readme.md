# Car Damage Assessment + RAG System

## Overview
A production-ready system that combines:
- **YOLOv11** for car part detection and damage assessment
- **RAG (Retrieval-Augmented Generation)** with Suzuki Swift manual
- **FastAPI** for unified API

## Architecture
- **Detection**: YOLOv11 (custom trained on Pakistani cars)
- **Embeddings**: bge-m3 (Ollama)
- **LLM**: Groq (openai/gpt-oss-20b)
- **Vector Store**: ChromaDB
- **API**: FastAPI with 4 workers

## Features
- [x] Car part detection (bonnet, bumper, headlights, etc.)
- [x] Damage assessment (scratches, dents, missing parts)
- [x] License plate recognition
- [x] RAG on Suzuki Swift repair manual
- [x] Combined inspection + repair lookup

## API Endpoints
- `/inspect` - YOLO-only inspection
- `/ask_manual` - Query the Suzuki Swift manual
- `/inspect_with_repair` - Combined detection + repair lookup
- `/health` - System health check

## Performance
- Detection: ~500ms per image
- RAG query: ~2-3 seconds
- Memory usage: ~6GB (with 2 workers)

## Deployment
- Local: FastAPI on 0.0.0.0:8000
- Production: Docker + Render.com (coming soon)