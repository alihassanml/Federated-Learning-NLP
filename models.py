"""
Model definitions for Federated RAG System
"""
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import get_peft_model, LoraConfig, TaskType
import os

class RetrieverModel:
    """Embedding model for document retrieval"""
    
    # Available models (ranked by quality)
    AVAILABLE_MODELS = {
        'mpnet': 'sentence-transformers/all-mpnet-base-v2',  # Best quality, 110M params
        'minilm': 'sentence-transformers/all-MiniLM-L6-v2',  # Fast, 23M params
        'multilingual': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'  # Multilingual, 278M params
    }
    
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Loaded retriever: {model_name} (dim={self.embedding_dim})")
    
    def encode(self, texts, batch_size=32):
        """Encode texts to embeddings"""
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    
    def get_state_dict(self):
        """Get model state dict for federated learning"""
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load model state dict"""
        self.model.load_state_dict(state_dict)
    
    def save(self, path):
        """Save model"""
        os.makedirs(path, exist_ok=True)
        self.model.save(path)
    
    def load(self, path):
        """Load model"""
        self.model = SentenceTransformer(path)


class GeneratorModel:
    """Generator model with LoRA for answer generation"""
    
    # Available generator models
    AVAILABLE_MODELS = {
        'flan-t5-small': 'google/flan-t5-small',    # 80M params
        'flan-t5-base': 'google/flan-t5-base',      # 250M params
        'flan-t5-large': 'google/flan-t5-large'     # 780M params
    }
    
    def __init__(self, model_name='google/flan-t5-base', use_lora=True):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        print(f"Loading generator: {model_name}")
        
        if use_lora:
            try:
                # Configure LoRA for T5 - target all attention projection layers
                lora_config = LoraConfig(
                    task_type=TaskType.SEQ_2_SEQ_LM,
                    r=8,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    target_modules=["q", "k", "v", "o"],  # All attention projections
                    inference_mode=False,
                    bias="none"
                )
                self.model = get_peft_model(self.model, lora_config)
                self.model.print_trainable_parameters()
                
                # Verify trainable parameters
                trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                if trainable_count == 0:
                    print("Warning: LoRA resulted in 0 trainable parameters. Using full model.")
                    # Reinitialize without LoRA
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                    # Freeze most layers, only train last few
                    for param in self.model.parameters():
                        param.requires_grad = False
                    # Unfreeze decoder
                    for param in self.model.decoder.parameters():
                        param.requires_grad = True
                else:
                    print(f"LoRA successfully configured with {trainable_count:,} trainable parameters")
                    
            except Exception as e:
                print(f"LoRA setup failed: {e}. Using partial model training.")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                # Freeze encoder, train decoder only
                for param in self.model.encoder.parameters():
                    param.requires_grad = False
                for param in self.model.decoder.parameters():
                    param.requires_grad = True
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def generate(self, input_text, max_length=128):
        """Generate answer from input text"""
        inputs = self.tokenizer(input_text, return_tensors='pt', 
                               max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def get_trainable_state_dict(self):
        """Get only trainable parameters (LoRA adapters)"""
        state_dict = {}
        
        # Get all state dict items
        full_state = self.model.state_dict()
        
        # Filter for LoRA parameters or trainable parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name in full_state:
                    state_dict[name] = full_state[name].cpu().clone()
        
        # If no trainable params found, return base model params (fallback)
        if not state_dict:
            print("Warning: No trainable parameters found. Using all parameters.")
            state_dict = {k: v.cpu().clone() for k, v in full_state.items()}
        
        return state_dict
    
    def load_adapter_state_dict(self, state_dict):
        """Load adapter state dict"""
        self.model.load_state_dict(state_dict, strict=False)
    
    def save(self, path):
        """Save model"""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def load(self, path):
        """Load model"""
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)


def aggregate_model_weights(client_updates, aggregation_method='fedavg'):
    if not client_updates:
        return None
    
    aggregated = {}
    total_samples = sum(u.get('num_samples', 1) for u in client_updates)
    
    for key in client_updates[0]['generator_updates'].keys():
        weighted_sum = 0
        for u in client_updates:
            weight = u['num_samples'] / total_samples
            weighted_sum = weighted_sum + u['generator_updates'][key].float() * weight if key in u['generator_updates'] else weighted_sum
        aggregated[key] = weighted_sum
    
    return aggregated



def add_differential_privacy_noise(state_dict, noise_multiplier=0.1, clip_norm=1.0):
    """
    Add differential privacy noise to model updates
    
    Args:
        state_dict: Model state dictionary
        noise_multiplier: Scale of Gaussian noise
        clip_norm: Gradient clipping norm
    
    Returns:
        State dict with DP noise added
    """
    noisy_state_dict = {}
    
    for key, value in state_dict.items():
        # Clip gradients
        norm = torch.norm(value)
        if norm > clip_norm:
            value = value * (clip_norm / norm)
        
        # Add Gaussian noise
        noise = torch.randn_like(value) * noise_multiplier
        noisy_state_dict[key] = value + noise
    
    return noisy_state_dict