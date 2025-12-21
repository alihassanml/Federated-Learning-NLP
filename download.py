# preload_models.py
"""
Pre-downloads all retriever and generator models to the default Hugging Face cache
so FastAPI app can use them directly without downloading at runtime.
"""

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Define models to preload
RETRIEVER_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
]

GENERATOR_MODELS = [
    "google/flan-t5-small",
    "google/flan-t5-base",
    "google/flan-t5-large"
]

def preload_sentence_transformers():
    print("Downloading SentenceTransformer models to default cache...")
    for model_name in RETRIEVER_MODELS:
        print(f"Downloading {model_name}...")
        # Use default cache by removing cache_folder
        SentenceTransformer(model_name)
    print("Retriever models downloaded successfully!\n")

def preload_transformers():
    print("Downloading Hugging Face generator models to default cache...")
    for model_name in GENERATOR_MODELS:
        print(f"Downloading {model_name}...")
        AutoTokenizer.from_pretrained(model_name)
        AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("Generator models downloaded successfully!\n")

if __name__ == "__main__":
    preload_sentence_transformers()
    preload_transformers()
    print("All models are preloaded and ready for the FastAPI app!")
