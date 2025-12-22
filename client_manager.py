"""
Client Manager for Federated Learning
"""
import torch
import copy
from rag_pipeline import RAGPipeline
from models import add_differential_privacy_noise
import json
import os
import base64
from typing import Dict, List, Tuple, Optional


class FederatedClient:
    """Represents a client (company) in federated learning"""
    
    def __init__(self, client_id, data_folder, retriever_model='sentence-transformers/all-mpnet-base-v2',
                 generator_model='google/flan-t5-base',use_lora=True):
        self.client_id = client_id
        self.data_folder = data_folder
        self.use_lora = use_lora
        self.rag_pipeline = RAGPipeline(client_id, data_folder, retriever_model, generator_model,use_lora=use_lora)
        self.training_history = []
        self.is_ready = False
        self.retriever_model = retriever_model
        self.generator_model = generator_model
        

        self.public_key = None
        self.private_key = None
        self.session_key = None
        self.peer_public_keys = {}
    
    def initialize(self):
        """Initialize client by loading and indexing documents"""
        try:
            num_chunks = self.rag_pipeline.load_and_index_documents()
            self.is_ready = True
            return {
                'status': 'success',
                'num_chunks': num_chunks,
                'message': f'Client {self.client_id} initialized with {num_chunks} chunks'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    

    def setup_secure_aggregation(self, keys_info: Dict):
        """
        Setup secure aggregation for this client
        
        Args:
            keys_info: Dict with public_key, private_key, session_key, all_public_keys
        """
        self.public_key = base64.b64decode(keys_info['public_key'])
        self.private_key = base64.b64decode(keys_info['private_key'])
        self.session_key = base64.b64decode(keys_info['session_key'])
        
        # Store peer public keys
        self.peer_public_keys = {
            cid: base64.b64decode(pk)
            for cid, pk in keys_info.get('all_public_keys', {}).items()
        }
        
        print(f"  [Secure] {self.client_id} configured for secure aggregation")



    def update_global_model(self, global_retriever_state=None, global_generator_state=None):
        """Update local model with global model weights"""
        try:
            if global_retriever_state:
                self.rag_pipeline.retriever.load_state_dict(global_retriever_state)
            
            if global_generator_state:
                self.rag_pipeline.generator.load_adapter_state_dict(global_generator_state)
            
            return {'status': 'success', 'message': 'Model updated'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def local_training(self, training_config):
        if not self.is_ready:
            return {'status': 'error', 'message': 'Client not initialized'}

        try:
            questions = training_config.get('questions', [])
            answers = training_config.get('answers', [])
            learning_rate = training_config.get('learning_rate', 1e-4)
            epochs = training_config.get('epochs', 1)
            use_dp = training_config.get('use_dp', True)
            dp_noise = training_config.get('dp_noise_multiplier', 0.1)

            if not questions or not answers:
                questions, answers = self._generate_default_training_data()

            print(f"\n{self.client_id}: Starting local training...")
            print(f"Training samples: {len(questions)}")

            epoch_losses = []

            for epoch in range(epochs):
                loss = self.rag_pipeline.train_step(
                    questions=questions,
                    answers=answers,
                    learning_rate=learning_rate,
                    epochs=1,  # train one epoch at a time
                    dp_noise=dp_noise if use_dp else 0.0
                )
                epoch_losses.append(loss)
                print(f"   Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

            # GET UPDATED LORA
            generator_updates = self.rag_pipeline.generator.get_trainable_state_dict()

            # Create masked update if secure aggregation enabled
            masked_update = None
            if hasattr(self, 'secure_aggregator') and self.secure_aggregator:
                active_clients = training_config.get('active_clients', [])
                masked_update = self.secure_aggregator.create_masked_update(
                    self.client_id,
                    generator_updates,
                    active_clients
                )
                print(f"  [Secure] {self.client_id} created masked update")

            # APPLY DP TO MODEL UPDATES (POST-TRAIN)
            if use_dp:
                generator_updates = add_differential_privacy_noise(
                    generator_updates,
                    noise_multiplier=dp_noise
                )
                print(f"{self.client_id}: Applied differential privacy")

            # RECORD TRAINING HISTORY
            self.training_history.append({
                'loss': epoch_losses[-1],
                'num_samples': len(questions),
                'epoch_losses': epoch_losses
            })

            return {
                'status': 'success',
                'client_id': self.client_id,
                'loss': epoch_losses[-1],
                'epoch_losses': epoch_losses,  # send back full epoch info
                'num_samples': len(questions),
                'generator_updates': generator_updates,
                'masked_update': masked_update,
                'message': f'Training completed. Loss: {epoch_losses[-1]:.4f}'
            }

        except Exception as e:
            print(f"Error in training: {e}")
            return {'status': 'error', 'message': str(e)}


    
    def _generate_default_training_data(self):
        """Generate default QA pairs from documents"""
        questions = []
        answers = []
        
        if self.client_id == "company1":
            questions = [
                "How many days of paid leave do employees get?",
                "What are the standard working hours?",
                "How many days can employees work remotely?",
                "How much sick leave is provided?"
            ]
            answers = [
                "Employees get 20 days of paid leave per year.",
                "Standard working hours are 9 AM to 5 PM, Monday through Friday.",
                "Employees may work remotely up to 3 days per week with manager approval.",
                "Sick leave provides up to 10 days annually."
            ]
        else:
            questions = [
                "How do I access VPN?",
                "What is the password policy?",
                "How often do passwords expire?",
                "How can I contact IT support?"
            ]
            answers = [
                "Use the Cisco AnyConnect VPN client with your email credentials.",
                "Passwords must be at least 12 characters with uppercase, lowercase, numbers, and special characters.",
                "Passwords expire every 90 days.",
                "Email support@company.com or call extension 5555."
            ]
        
        return questions, answers
    
    def query(self, question, top_k=3):
        """Query the RAG system"""
        if not self.is_ready:
            return {'status': 'error', 'message': 'Client not initialized'}
        
        try:
            result = self.rag_pipeline.query(question, top_k=top_k)
            return {
                'status': 'success',
                'answer': result['answer'],
                'sources': result['sources']
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def get_status(self):
        """Get client status"""
        return {
            'client_id': self.client_id,
            'is_ready': self.is_ready,
            'num_documents': len(self.rag_pipeline.chunks),
            'training_rounds': len(self.training_history),
            'retriever_model': self.retriever_model,
            'generator_model': self.generator_model
        }
    
    def save_model(self):
        """Save client model"""
        save_dir = f"models/{self.client_id}"
        os.makedirs(save_dir, exist_ok=True)
        
        self.rag_pipeline.retriever.save(os.path.join(save_dir, 'retriever'))
        self.rag_pipeline.generator.save(os.path.join(save_dir, 'generator'))
        
        return {'status': 'success', 'message': 'Model saved'}


class ClientManager:
    """Manages multiple federated clients"""
    
    def __init__(self):
        self.clients = {}
    
    def register_client(self, client_id, data_folder, retriever_model='sentence-transformers/all-mpnet-base-v2',
                       generator_model='google/flan-t5-base',use_lora=True):
        """Register a new client"""
        if client_id in self.clients:
            return {'status': 'error', 'message': 'Client already registered'}
        
        client = FederatedClient(client_id, data_folder, retriever_model, generator_model,use_lora=use_lora)
        self.clients[client_id] = client
        
        return {'status': 'success', 'message': f'Client {client_id} registered'}
    
    def initialize_client(self, client_id):
        """Initialize a client"""
        if client_id not in self.clients:
            return {'status': 'error', 'message': 'Client not found'}
        
        return self.clients[client_id].initialize()
    
    def get_client(self, client_id):
        """Get a client"""
        return self.clients.get(client_id)
    
    def get_all_clients(self):
        """Get all clients"""
        return self.clients
    
    def get_client_status(self, client_id):
        """Get client status"""
        if client_id not in self.clients:
            return {'status': 'error', 'message': 'Client not found'}
        
        return self.clients[client_id].get_status()
    
    def get_all_status(self):
        """Get status of all clients"""
        return {
            client_id: client.get_status()
            for client_id, client in self.clients.items()
        }