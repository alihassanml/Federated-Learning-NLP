from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
from datetime import datetime

from client_manager import ClientManager
from federated_server import FederatedServer

# Initialize FastAPI
app = FastAPI(title="FedSearch-NLP", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates
templates = Jinja2Templates(directory="frontend/templates")

# Initialize managers
client_manager = ClientManager()
fed_server = FederatedServer()

# Create data directories
os.makedirs("data/company1", exist_ok=True)
os.makedirs("data/company2", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Pydantic models
class TrainingConfig(BaseModel):
    learning_rate: float = 1e-4
    epochs: int = 1
    use_dp: bool = True
    dp_noise_multiplier: float = 0.1
    aggregation_method: str = "fedavg"
    questions: Optional[List[str]] = None
    answers: Optional[List[str]] = None

class InitializeRequest(BaseModel):
    retriever_model: str = "sentence-transformers/all-mpnet-base-v2"
    generator_model: str = "google/flan-t5-base"

class QueryRequest(BaseModel):
    question: str
    company_id: str
    top_k: int = 3

# Routes

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve main UI"""
    with open("frontend/templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/available-models")
async def get_available_models():
    """Get list of available models"""
    return {
        "retriever_models": [
            {
                "id": "minilm",
                "name": "all-MiniLM-L6-v2",
                "full_name": "sentence-transformers/all-MiniLM-L6-v2",
                "parameters": "23M",
                "embedding_dim": 384,
                "speed": "Fast",
                "quality": "⭐⭐⭐⭐",
                "recommended": False
            },
            {
                "id": "mpnet",
                "name": "all-mpnet-base-v2",
                "full_name": "sentence-transformers/all-mpnet-base-v2",
                "parameters": "110M",
                "embedding_dim": 768,
                "speed": "Medium",
                "quality": "⭐⭐⭐⭐⭐",
                "recommended": True
            },
            {
                "id": "multilingual",
                "name": "paraphrase-multilingual-mpnet",
                "full_name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                "parameters": "278M",
                "embedding_dim": 768,
                "speed": "Slow",
                "quality": "⭐⭐⭐⭐⭐",
                "recommended": False
            },
            
            
        ],
        "generator_models": [
            {
                "id": "flan-t5-small",
                "name": "Flan-T5-Small",
                "full_name": "google/flan-t5-small",
                "parameters": "80M",
                "speed": "Fast",
                "quality": "⭐⭐⭐⭐",
                "recommended": False
            },
            {
                "id": "flan-t5-base",
                "name": "Flan-T5-Base",
                "full_name": "google/flan-t5-base",
                "parameters": "250M",
                "speed": "Medium",
                "quality": "⭐⭐⭐⭐⭐",
                "recommended": True
            },
            {
                "id": "flan-t5-large",
                "name": "Flan-T5-Large",
                "full_name": "google/flan-t5-large",
                "parameters": "780M",
                "speed": "Slow",
                "quality": "⭐⭐⭐⭐⭐",
                "recommended": False
            }
        ]
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.post("/api/initialize")
async def initialize_system(config: InitializeRequest):
    """Initialize the federated system"""
    try:
        # Register clients with selected models
        client_manager.register_client("company1", "data/company1", 
                                      config.retriever_model, config.generator_model)
        client_manager.register_client("company2", "data/company2",
                                      config.retriever_model, config.generator_model)
        
        # Initialize global model
        fed_result = fed_server.initialize_global_model()
        
        return {
            "status": "success",
            "message": "System initialized",
            "retriever_model": config.retriever_model,
            "generator_model": config.generator_model,
            "details": fed_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-documents/{company_id}")
async def upload_documents(
    company_id: str,
    files: List[UploadFile] = File(...)
):
    """Upload documents for a company"""
    try:
        if company_id not in ["company1", "company2"]:
            raise HTTPException(status_code=400, detail="Invalid company ID")
        
        data_folder = f"data/{company_id}"
        os.makedirs(data_folder, exist_ok=True)
        
        uploaded_files = []
        
        for file in files:
            file_path = os.path.join(data_folder, file.filename)
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            uploaded_files.append(file.filename)
        
        return {
            "status": "success",
            "company_id": company_id,
            "uploaded_files": uploaded_files,
            "message": f"Uploaded {len(uploaded_files)} files for {company_id}"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/initialize-client/{company_id}")
async def initialize_client(company_id: str):
    """Initialize a client (load and index documents)"""
    try:
        if company_id not in ["company1", "company2"]:
            raise HTTPException(status_code=400, detail="Invalid company ID")
        
        # Register if not already registered
        if company_id not in client_manager.clients:
            client_manager.register_client(company_id, f"data/{company_id}")
        
        # Initialize
        result = client_manager.initialize_client(company_id)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train")
async def start_training(config: TrainingConfig):
    """Start federated training round"""
    try:
        # Ensure clients are initialized
        for client_id in ["company1", "company2"]:
            if client_id not in client_manager.clients:
                client_manager.register_client(client_id, f"data/{client_id}")
            
            client = client_manager.get_client(client_id)
            if not client.is_ready:
                init_result = client.initialize()
                if init_result['status'] != 'success':
                    return {
                        "status": "error",
                        "message": f"Failed to initialize {client_id}: {init_result['message']}"
                    }
        
        # Convert config to dict
        training_config = config.model_dump()
        
        # Execute federated training round
        result = fed_server.federated_training_round(
            client_manager.get_all_clients(),
            training_config
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def query_system(request: QueryRequest):
    """Query the RAG system"""
    try:
        client = client_manager.get_client(request.company_id)
        
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        
        if not client.is_ready:
            raise HTTPException(status_code=400, detail="Client not initialized")
        
        result = client.query(request.question, top_k=request.top_k)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_status():
    """Get system status"""
    try:
        client_status = client_manager.get_all_status()
        training_history = fed_server.get_training_history()
        
        return {
            "status": "success",
            "server_initialized": fed_server.is_initialized,
            "current_round": fed_server.round_number,
            "clients": client_status,
            "training_history": training_history
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/training-history")
async def get_training_history():
    """Get detailed training history"""
    try:
        return fed_server.get_training_history()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/save-model")
async def save_model():
    """Save global model"""
    try:
        result = fed_server.save_global_model()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/load-model")
async def load_model():
    """Load global model"""
    try:
        result = fed_server.load_global_model()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/reset")
async def reset_system():
    """Reset the entire system"""
    try:
        # Clear data folders
        for company_id in ["company1", "company2"]:
            folder = f"data/{company_id}"
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    os.remove(os.path.join(folder, file))
        
        # Reinitialize
        global client_manager, fed_server
        client_manager = ClientManager()
        fed_server = FederatedServer()
        
        return {
            "status": "success",
            "message": "System reset successfully"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)