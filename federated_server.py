"""
Federated Learning Server for Model Aggregation
"""
import torch
import copy
from models import aggregate_model_weights, RetrieverModel, GeneratorModel
import json
from datetime import datetime
from adaptive_privacy import AdaptivePrivacy
from byzantine_defense import ByzantineDefense
from typing import List, Dict, Tuple
from secure_aggregation import SecureAggregator, SecureChannel
import base64



class FederatedServer:
    """Central server for federated learning aggregation"""
    
    def __init__(self):
        self.byzantine_defense = ByzantineDefense(...)

        self.global_retriever = None
        self.global_generator = None
        self.round_number = 0
        self.training_history = []
        self.is_initialized = False
        self.privacy_controller = AdaptivePrivacy(base_noise=0.1, min_noise=0.01, decay=0.95)

        self.byzantine_defense = ByzantineDefense(
            method='norm_filter',  # Options: 'krum', 'median', 'trimmed_mean', 'norm_filter'
            detection_threshold=2.5
        )

        self.use_secure_aggregation = False
        self.secure_aggregator = None
        self.secure_channel = SecureChannel()


    
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
    
    def aggregate_client_updates(self, client_updates, aggregation_method='fedavg', 
                            use_byzantine_defense=True):
        """
        Aggregate updates from multiple clients with Byzantine robustness
        
        Args:
            client_updates: List of dicts returned by clients
            aggregation_method: 'fedavg' or 'fedprox'
            use_byzantine_defense: Whether to apply Byzantine defense
        """
        if not self.is_initialized:
            return {'status': 'error', 'message': 'Server not initialized'}

        if not client_updates:
            return {'status': 'error', 'message': 'No client updates provided'}

        try:
            print(f"\n=== Federated Aggregation Round {self.round_number + 1} ===")
            print(f"Received updates from {len(client_updates)} clients")
            print(f"Byzantine Defense: {'Enabled' if use_byzantine_defense else 'Disabled'}")

            # Filter VALID client updates
            valid_clients = []
            generator_updates = []
            client_weights = []

            for update in client_updates:
                if (
                    'generator_updates' not in update
                    or update['generator_updates'] is None
                    or len(update['generator_updates']) == 0
                ):
                    print(f"Skipping client {update.get('client_id', 'unknown')} — missing updates")
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

            # ============ BYZANTINE DEFENSE ============
            rejected_clients = []
            
            # ============ SECURE AGGREGATION ============
            if self.use_secure_aggregation and self.secure_aggregator:
                print("  [Secure Aggregation] Processing encrypted updates...")
                
                # Extract masked updates
                masked_updates = []
                client_ids = []
                
                for update in valid_clients:
                    if 'masked_update' in update and update['masked_update'] is not None:
                        masked_updates.append(update['masked_update'])
                        client_ids.append(update['client_id'])
                    else:
                        # Client didn't mask - add to masked list
                        masked_update = self.secure_aggregator.create_masked_update(
                            update['client_id'],
                            update['generator_updates'],
                            [u['client_id'] for u in valid_clients]
                        )
                        masked_updates.append(update['generator_updates'])
                        client_ids.append(update['client_id'])
                
                # Aggregate masked updates (masks cancel out)
                aggregated_state = self.secure_aggregator.aggregate_masked_updates(
                    masked_updates,
                    client_ids
                )
                
                # Increment round for new masks
                self.secure_aggregator.increment_round()
                
                print(f"  [Privacy] Individual updates never exposed to server!")

            elif use_byzantine_defense:
                # Use Byzantine-robust aggregation
                aggregated_state, rejected_clients = self.byzantine_defense.aggregate_updates(
                    valid_clients,
                    client_weights
                )
                
                # Log rejected clients
                if rejected_clients:
                    print(f"⚠ Byzantine Defense rejected clients: {rejected_clients}")
                    for client_id in rejected_clients:
                        rep = self.byzantine_defense.get_client_reputation(client_id)
                        print(f"   {client_id} reputation: {rep:.2f}")
            else:
                # Standard weighted FedAvg (no defense)
                total_samples = sum(client_weights)
                aggregated_state = {}

                for key in generator_updates[0].keys():
                    aggregated_state[key] = torch.zeros_like(generator_updates[0][key])

                    for client_state, weight in zip(generator_updates, client_weights):
                        aggregated_state[key] += client_state[key] * (weight / total_samples)
            
            # Update global generator
            self.global_generator.load_adapter_state_dict(aggregated_state)

            # Metrics
            avg_loss = sum(u['loss'] * u['num_samples'] for u in valid_clients) / sum(client_weights)

            round_info = {
                'round': self.round_number + 1,
                'timestamp': datetime.now().isoformat(),
                'num_clients': len(valid_clients),
                'rejected_clients': rejected_clients,
                'avg_loss': avg_loss,
                'total_samples': sum(client_weights),
                'aggregation_method': aggregation_method,
                'byzantine_defense': use_byzantine_defense
            }

            self.training_history.append(round_info)
            self.round_number += 1

            print(f"Round {self.round_number} aggregation successful")
            print(f"Weighted avg loss: {avg_loss:.4f}")
            print(f"Rejected clients: {len(rejected_clients)}")

            return {
                'status': 'success',
                'round': self.round_number,
                'avg_loss': avg_loss,
                'total_samples': sum(client_weights),
                'rejected_clients': rejected_clients,
                'message': f'Round {self.round_number} aggregation successful'
            }

        except Exception as e:
            print(f"Error in aggregation: {e}")
            return {'status': 'error', 'message': str(e)}

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
    
    def get_byzantine_stats(self) -> Dict:
        """Get Byzantine defense statistics"""
        return self.byzantine_defense.get_defense_stats()

    def exclude_suspicious_clients(self, threshold: int = 3) -> List[str]:
        """
        Get list of clients that should be excluded
        
        Args:
            threshold: Number of suspicious rounds before exclusion
            
        Returns:
            List of client IDs to exclude
        """
        excluded = []
        for client_id in self.byzantine_defense.suspicious_clients.keys():
            if self.byzantine_defense.should_exclude_client(client_id, threshold):
                excluded.append(client_id)
        
        return excluded
    
    def enable_secure_aggregation(self, num_clients: int):
        """
        Enable secure aggregation protocol
        
        Args:
            num_clients: Expected number of clients
        """
        self.secure_aggregator = SecureAggregator(num_clients=num_clients)
        self.use_secure_aggregation = True
        
        print(f"✓ Secure Aggregation enabled for {num_clients} clients")
        print(f"  Protocol: Pairwise masking with cryptographic key exchange")
        
        return {
            'status': 'success',
            'message': 'Secure aggregation enabled',
            'num_clients': num_clients
        }
    
    def setup_secure_clients(self, client_ids: List[str]) -> Dict:
        """
        Setup secure aggregation for clients
        
        Args:
            client_ids: List of participating client IDs
            
        Returns:
            Dict mapping client_id to their keys
        """
        if not self.use_secure_aggregation:
            return {}
        
        client_keys = {}
        
        for client_id in client_ids:
            public_key, private_key = self.secure_aggregator.generate_client_keys(client_id)
            session_key = self.secure_channel.establish_session(client_id)
            
            client_keys[client_id] = {
                'public_key': base64.b64encode(public_key).decode(),
                'private_key': base64.b64encode(private_key).decode(),
                'session_key': base64.b64encode(session_key).decode()
            }
        
        # Broadcast public keys to all clients
        public_keys = self.secure_aggregator.broadcast_public_keys()
        
        return {
            'client_keys': client_keys,
            'all_public_keys': {
                cid: base64.b64encode(pk).decode() 
                for cid, pk in public_keys.items()
            }
        }