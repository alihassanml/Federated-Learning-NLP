"""
Federated Learning Server for Model Aggregation
"""
import torch
import copy
from models import aggregate_model_weights, RetrieverModel, GeneratorModel
import json
from datetime import datetime
from adaptive_privacy import AdaptivePrivacy



class FederatedServer:
    """Central server for federated learning aggregation"""
    
    def __init__(self):
        self.global_retriever = None
        self.global_generator = None
        self.round_number = 0
        self.training_history = []
        self.is_initialized = False
        self.privacy_controller = AdaptivePrivacy(base_noise=0.1, min_noise=0.01, decay=0.95)

    
    def initialize_global_model(self):
        print("Initializing global models...")
        
        self.global_retriever = RetrieverModel()
        self.global_generator = GeneratorModel(use_lora=True)

        self.is_initialized = True

        return {
            'status': 'success',
            'message': 'Global models initialized',
            'retriever_model': self.global_retriever.model_name,
            'generator_model': self.global_generator.model_name
        }

    
    def get_global_model_state(self):
        """Get current global model state"""
        if not self.is_initialized:
            return None, None
        
        return (
            self.global_retriever.get_state_dict(),
            self.global_generator.get_trainable_state_dict()
        )
    
    def aggregate_client_updates(self, client_updates, aggregation_method='fedavg'):
        """
        Aggregate updates from multiple clients (Weighted FedAvg / FedProx safe)

        Args:
            client_updates: List of dicts returned by clients
            aggregation_method: 'fedavg' or 'fedprox'
        """
        if not self.is_initialized:
            return {'status': 'error', 'message': 'Server not initialized'}

        if not client_updates:
            return {'status': 'error', 'message': 'No client updates provided'}

        try:
            print(f"\n=== Federated Aggregation Round {self.round_number + 1} ===")
            print(f"Received updates from {len(client_updates)} clients")

            # --------------------------------------------------
            # 1️⃣ Filter VALID client updates only
            # --------------------------------------------------
            valid_clients = []
            generator_updates = []
            client_weights = []

            for update in client_updates:
                if (
                    'generator_updates' not in update
                    or update['generator_updates'] is None
                    or len(update['generator_updates']) == 0
                ):
                    print(
                        f"Skipping client {update.get('client_id', 'unknown')} "
                        f"— missing generator_updates"
                    )
                    continue

                valid_clients.append(update)
                generator_updates.append(update['generator_updates'])
                client_weights.append(update.get('num_samples', 1))

            if not generator_updates:
                return {
                    'status': 'error',
                    'message': 'No valid generator updates to aggregate'
                }

            print(f"Valid clients for aggregation: {len(generator_updates)}")

            # --------------------------------------------------
            # 2️⃣ Weighted Federated Averaging
            # --------------------------------------------------
            total_samples = sum(client_weights)
            aggregated_state = {}

            for key in generator_updates[0].keys():
                aggregated_state[key] = torch.zeros_like(
                    generator_updates[0][key]
                )

                for client_state, weight in zip(generator_updates, client_weights):
                    aggregated_state[key] += (
                        client_state[key] * (weight / total_samples)
                    )

            # --------------------------------------------------
            # 3️⃣ Update global generator (LoRA adapters)
            # --------------------------------------------------
            self.global_generator.load_adapter_state_dict(aggregated_state)

            # --------------------------------------------------
            # 4️⃣ Metrics & bookkeeping
            # --------------------------------------------------
            avg_loss = sum(
                u['loss'] * u['num_samples'] for u in valid_clients
            ) / total_samples

            round_info = {
                'round': self.round_number + 1,
                'timestamp': datetime.now().isoformat(),
                'num_clients': len(valid_clients),
                'avg_loss': avg_loss,
                'total_samples': total_samples,
                'aggregation_method': aggregation_method
            }

            self.training_history.append(round_info)
            self.round_number += 1

            print(f"Round {self.round_number} aggregation successful")
            print(f"Weighted avg loss: {avg_loss:.4f}")
            print(f"Total samples: {total_samples}")

            return {
                'status': 'success',
                'round': self.round_number,
                'avg_loss': avg_loss,
                'total_samples': total_samples,
                'message': f'Round {self.round_number} aggregation successful'
            }

        except Exception as e:
            print(f"Error in aggregation: {e}")
            return {'status': 'error', 'message': str(e)}

    
    def federated_training_round(self, clients, training_config):
        """
        Execute one complete federated training round
        
        Args:
            clients: Dict of FederatedClient objects
            training_config: Training configuration
        
        Returns:
            Dict with round results
        """
        if not self.is_initialized:
            self.initialize_global_model()
        
        print(f"\n{'='*60}")
        print(f"Starting Federated Learning Round {self.round_number + 1}")
        print(f"{'='*60}")
        
        # Get global model state
        global_retriever_state, global_generator_state = self.get_global_model_state()
        
        # Distribute global model to clients
        print("\n1. Distributing global model to clients...")
        for client_id, client in clients.items():
            if client.is_ready:
                client.update_global_model(global_retriever_state, global_generator_state)
                print(f"   ✓ {client_id} updated")
        
        # Local training on each client
        print("\n2. Local training on clients...")
        client_updates = []

        adaptive_noise = self.privacy_controller.get_noise_multiplier(self.round_number)
        training_config["dp_noise_multiplier"] = adaptive_noise
        print(f"Adaptive DP Noise for round {self.round_number + 1}: {adaptive_noise}")
        
        for client_id, client in clients.items():
            if client.is_ready:
                print(f"\n   Training {client_id}...")
                result = client.local_training(training_config)
                
                if result['status'] == 'success':
                    client_updates.append(result)
                    print(f"   ✓ {client_id}: Loss = {result['loss']:.4f}")
                else:
                    print(f"   ✗ {client_id}: {result['message']}")
        
        # Aggregate updates
        print("\n3. Aggregating client updates...")
        aggregation_result = self.aggregate_client_updates(
            client_updates,
            aggregation_method=training_config.get('aggregation_method', 'fedavg')
        )
        
        if aggregation_result['status'] == 'success':
            avg_loss = aggregation_result['avg_loss']
            self.privacy_controller.update_loss(avg_loss)   # <--- IMPORTANT
            print(f"[Adaptive DP] Updated noise schedule using avg_loss={avg_loss}")
        else:
            print(f"   ✗ Aggregation failed: {aggregation_result['message']}")
        
        print(f"\n{'='*60}")
        print(f"Round {self.round_number} Complete")
        print(f"{'='*60}\n")
        
        return aggregation_result
    
    def get_training_history(self):
        """Get training history"""
        return {
            'current_round': self.round_number,
            'total_rounds': len(self.training_history),
            'history': self.training_history
        }
    
    def save_global_model(self, save_dir='models/global'):
        """Save global model"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        self.global_retriever.save(os.path.join(save_dir, 'retriever'))
        self.global_generator.save(os.path.join(save_dir, 'generator'))
        
        # Save training history
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        return {'status': 'success', 'message': 'Global model saved'}
    
    def load_global_model(self, load_dir='models/global'):
        """Load global model"""
        import os
        
        if not os.path.exists(load_dir):
            return {'status': 'error', 'message': 'Model directory not found'}
        
        self.global_retriever = RetrieverModel()
        self.global_generator = GeneratorModel(use_lora=True)
        
        self.global_retriever.load(os.path.join(load_dir, 'retriever'))
        self.global_generator.load(os.path.join(load_dir, 'generator'))
        
        # Load training history
        history_path = os.path.join(load_dir, 'training_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
                self.round_number = len(self.training_history)
        
        self.is_initialized = True
        
        return {'status': 'success', 'message': 'Global model loaded'}