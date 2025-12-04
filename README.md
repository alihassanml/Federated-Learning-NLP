# FedSearch-NLP: Federated RAG System

A complete federated learning system for training RAG (Retrieval-Augmented Generation) models across multiple organizations while keeping data private.

## 🌟 Features

- **Federated Learning**: Train models collaboratively without sharing raw data
- **Privacy-Preserving**: Differential privacy and encrypted updates
- **RAG Pipeline**: Complete retrieval and generation pipeline
- **Web Interface**: Easy-to-use HTML/Tailwind CSS frontend
- **Multi-Organization**: Support for multiple companies/departments
- **Real-time Updates**: Live training progress and status monitoring

## 📋 Requirements

- Python 3.8+
- Windows/Linux/Mac
- 4GB+ RAM
- (Optional) GPU for faster training

## 🚀 Installation

### 1. Create Project Structure

```bash
mkdir fedsearch-nlp
cd fedsearch-nlp
mkdir backend frontend data models logs
mkdir frontend/templates
mkdir data/company1 data/company2
```

### 2. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### 3. Download Models (First Run)

The models will be automatically downloaded on first use:
- `sentence-transformers/all-MiniLM-L6-v2` (~80MB)
- `google/flan-t5-small` (~300MB)

## 📁 File Structure

```
fedsearch-nlp/
├── backend/
│   ├── server.py                 # FastAPI main server
│   ├── federated_server.py       # FL aggregation logic
│   ├── client_manager.py         # Client training manager
│   ├── models.py                 # Model definitions
│   ├── rag_pipeline.py           # RAG implementation
│   ├── utils.py                  # Helper functions
│   └── requirements.txt          # Dependencies
├── frontend/
│   └── templates/
│       └── index.html            # Web UI
├── data/
│   ├── company1/                 # Company 1 documents
│   └── company2/                 # Company 2 documents
├── models/                       # Saved models
└── logs/                         # Training logs
```

## 🎯 Usage

### Step 1: Start the Server

```bash
cd backend
python server.py
```

The server will start at `http://localhost:8000`

### Step 2: Open Web Interface

Open your browser and navigate to:
```
http://localhost:8000
```

### Step 3: Initialize System

1. Click **"Initialize System"** to set up the federated server
2. This registers both companies and initializes global models

### Step 4: Upload Documents

**For Company 1:**
1. Click "Choose Files" under Company 1
2. Select PDF, TXT, or DOCX files
3. Click "Upload Documents"
4. Click "Initialize Client" to process documents

**For Company 2:**
1. Repeat the same process for Company 2

**Note**: If no documents are uploaded, sample documents will be created automatically.

### Step 5: Train the Model

1. Configure training parameters:
   - **Learning Rate**: 0.0001 (default)
   - **Epochs**: 1 (default)
   - **Differential Privacy**: Enabled (recommended)

2. Click **"Start Training Round"**

3. The system will:
   - Distribute global model to clients
   - Train locally on each company's data
   - Aggregate updates with privacy preservation
   - Update global model

4. View training progress and results

### Step 6: Query the System

1. Enter a question (e.g., "What is the leave policy?")
2. Select which company's data to query
3. Click "Ask Question"
4. View the answer and source documents

## 🔧 Configuration

### Training Parameters

```python
{
    "learning_rate": 1e-4,      # Learning rate for training
    "epochs": 1,                # Training epochs per round
    "use_dp": True,             # Enable differential privacy
    "dp_noise_multiplier": 0.1, # DP noise level
    "aggregation_method": "fedavg"  # FedAvg or FedProx
}
```

### Supported File Types

- **PDF**: `.pdf`
- **Text**: `.txt`
- **Word**: `.docx`

## 📊 How It Works

### Federated Learning Flow

1. **Initialization**
   - Global model created on server
   - Clients register with their data

2. **Training Round**
   ```
   Server → Distribute Global Model → Clients
   Clients → Local Training → Generate Updates
   Clients → Add DP Noise → Send Updates
   Server → Aggregate Updates → Update Global Model
   ```

3. **Privacy Protection**
   - Raw data never leaves client
   - Only model updates are shared
   - Differential privacy adds noise
   - Secure aggregation

### RAG Pipeline

1. **Indexing**
   - Documents chunked into passages
   - Embedded using sentence transformers
   - Stored in FAISS index

2. **Retrieval**
   - Query embedded
   - Top-k similar passages retrieved
   - Context gathered

3. **Generation**
   - Query + Context → Prompt
   - Flan-T5 generates answer
   - LoRA adapters for efficient training

## 🛠️ API Endpoints

- `GET /` - Web interface
- `POST /api/initialize` - Initialize system
- `POST /api/upload-documents/{company_id}` - Upload files
- `POST /api/initialize-client/{company_id}` - Initialize client
- `POST /api/train` - Start training round
- `POST /api/query` - Query RAG system
- `GET /api/status` - Get system status
- `GET /api/training-history` - Get training history

## 📈 Training Tips

1. **More Documents = Better Results**
   - Upload 5-10 documents per company
   - Diverse document types help

2. **Training Rounds**
   - Run 3-5 rounds for good convergence
   - Monitor loss reduction

3. **Differential Privacy**
   - Keep enabled for privacy
   - Adjust noise if needed

4. **Learning Rate**
   - Default (1e-4) works well
   - Reduce if training unstable

## 🐛 Troubleshooting

### Issue: Models not downloading
**Solution**: Ensure internet connection and run:
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

### Issue: CUDA out of memory
**Solution**: Training will automatically use CPU if GPU unavailable

### Issue: Upload fails
**Solution**: Check file permissions in `data/` folders

### Issue: Client initialization fails
**Solution**: Ensure documents are uploaded first or let system create sample data

## 📝 Example Questions to Try

**Company 1 (HR Policies):**
- "How many days of paid leave do employees get?"
- "What are the working hours?"
- "Can employees work remotely?"

**Company 2 (IT Systems):**
- "How do I access the VPN?"
- "What is the password policy?"
- "How can I contact IT support?"

## 🔐 Privacy Features

1. **No Raw Data Sharing**
   - Documents stay on local client
   - Only model parameters transmitted

2. **Differential Privacy**
   - Gaussian noise added to updates
   - Configurable privacy budget

3. **Secure Aggregation**
   - Updates aggregated securely
   - Individual contributions hidden

## 🚀 Advanced Usage

### Custom Training Data

Modify training questions/answers in `client_manager.py`:

```python
questions = ["Your custom question?"]
answers = ["Your custom answer."]
```

### Change Models

In `models.py`:
```python
# Use different models
retriever = RetrieverModel('sentence-transformers/all-mpnet-base-v2')
generator = GeneratorModel('google/flan-t5-base')
```

### Save/Load Models

```python
# Save
POST /api/save-model

# Load
POST /api/load-model
```

## 📚 References

- [Federated Learning](https://arxiv.org/abs/1602.05629)
- [Differential Privacy](https://arxiv.org/abs/1607.00133)
- [RAG](https://arxiv.org/abs/2005.11401)
- [LoRA](https://arxiv.org/abs/2106.09685)

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

MIT License - feel free to use for your projects!

## 🎉 Enjoy!

You now have a complete federated learning RAG system running on your machine!