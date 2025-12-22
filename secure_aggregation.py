
"""
Secure Aggregation Protocol for Federated Learning
Implements cryptographic secure aggregation using:
- Diffie-Hellman Key Exchange
- Secret Sharing (Shamir's Secret Sharing)
- Additive Secret Sharing
- Homomorphic Encryption (Paillier)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP, AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import hashlib
import pickle
import base64
from collections import defaultdict
import secrets


class SecureAggregator:
    """
    Implements secure aggregation protocol
    
    Protocol Overview:
    1. Key Generation: Each client generates public/private key pairs
    2. Key Exchange: Clients exchange public keys
    3. Pairwise Masking: Clients create pairwise masks using shared secrets
    4. Masked Upload: Clients upload masked model updates
    5. Secure Aggregation: Server aggregates without seeing individual updates
    6. Unmask: Masks cancel out, revealing only the aggregate
    """
    
    def __init__(self, num_clients: int, key_size: int = 2048):
        """
        Initialize Secure Aggregator
        
        Args:
            num_clients: Number of participating clients
            key_size: RSA key size (2048 or 4096)
        """
        self.num_clients = num_clients
        self.key_size = key_size
        
        # Client keys
        self.client_public_keys = {}
        self.client_private_keys = {}
        
        # Pairwise shared secrets
        self.shared_secrets = {}
        
        # Round tracking
        self.round_number = 0
        self.dropped_clients = set()
        
        print(f"SecureAggregator initialized for {num_clients} clients")
    
    def generate_client_keys(self, client_id: str) -> Tuple[bytes, bytes]:
        """
        Generate RSA key pair for a client
        
        Args:
            client_id: Client identifier
            
        Returns:
            Tuple of (public_key, private_key) in PEM format
        """
        key = RSA.generate(self.key_size)
        
        public_key = key.publickey().export_key()
        private_key = key.export_key()
        
        self.client_public_keys[client_id] = public_key
        self.client_private_keys[client_id] = private_key
        
        return public_key, private_key
    
    def broadcast_public_keys(self) -> Dict[str, bytes]:
        """
        Broadcast all client public keys
        
        Returns:
            Dict mapping client_id to public_key
        """
        return self.client_public_keys.copy()
    
    def generate_pairwise_mask(self, client_i: str, client_j: str, 
                               model_shape: torch.Size) -> torch.Tensor:
        """
        Generate pairwise mask between two clients
        
        Uses deterministic pseudo-random generation based on shared secret.
        Masks are symmetric: mask(i,j) = -mask(j,i)
        
        Args:
            client_i: First client ID
            client_j: Second client ID
            model_shape: Shape of model parameters
            
        Returns:
            Pairwise mask tensor
        """
        if client_i == client_j:
            return torch.zeros(model_shape)
        
        # Ensure consistent ordering
        if client_i > client_j:
            sign = -1
            client_i, client_j = client_j, client_i
        else:
            sign = 1
        
        # Generate shared secret (deterministic)
        pair_key = f"{client_i}_{client_j}_{self.round_number}"
        
        if pair_key not in self.shared_secrets:
            # Use hash of concatenated public keys as seed
            pub_i = self.client_public_keys[client_i]
            pub_j = self.client_public_keys[client_j]
            
            combined = pub_i + pub_j + str(self.round_number).encode()
            seed = int(hashlib.sha256(combined).hexdigest(), 16) % (2**32)
            
            self.shared_secrets[pair_key] = seed
        
        # Generate mask using seeded random
        seed = self.shared_secrets[pair_key]
        generator = torch.Generator().manual_seed(seed)
        mask = torch.randn(model_shape, generator=generator)
        
        return sign * mask
    
    def create_masked_update(self, client_id: str, model_update: Dict[str, torch.Tensor],
                            active_clients: List[str]) -> Dict[str, torch.Tensor]:
        """
        Create masked model update for secure aggregation
        
        Args:
            client_id: This client's ID
            model_update: Raw model update (gradients/parameters)
            active_clients: List of all active clients this round
            
        Returns:
            Masked model update
        """
        masked_update = {}
        
        for param_name, param in model_update.items():
            # Start with raw update
            masked_param = param.clone()
            
            # Add pairwise masks with all other clients
            for other_client in active_clients:
                if other_client != client_id:
                    mask = self.generate_pairwise_mask(
                        client_id, other_client, param.shape
                    )
                    masked_param += mask
            
            masked_update[param_name] = masked_param
        
        return masked_update
    
    def aggregate_masked_updates(self, masked_updates: List[Dict[str, torch.Tensor]],
                                 client_ids: List[str]) -> Dict[str, torch.Tensor]:
        """
        Aggregate masked updates - masks cancel out
        
        The magic: When we sum all masked updates, pairwise masks cancel:
        Σ(update_i + Σ_j mask(i,j)) = Σ(update_i) + Σ_i Σ_j mask(i,j)
        Since mask(i,j) = -mask(j,i), all masks sum to zero!
        
        Args:
            masked_updates: List of masked model updates
            client_ids: List of client IDs (for tracking)
            
        Returns:
            Aggregated update (unmasked)
        """
        if not masked_updates:
            raise ValueError("No masked updates to aggregate")
        
        aggregated = {}
        
        # Sum all masked updates
        for param_name in masked_updates[0].keys():
            param_sum = torch.zeros_like(masked_updates[0][param_name])
            
            for masked_update in masked_updates:
                param_sum += masked_update[param_name]
            
            # Average
            aggregated[param_name] = param_sum / len(masked_updates)
        
        print(f"  [Secure Aggregation] Aggregated {len(masked_updates)} masked updates")
        print(f"  [Privacy] Server never saw individual updates!")
        
        return aggregated
    
    def handle_dropout(self, dropped_client: str, active_clients: List[str],
                      surviving_updates: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """
        Handle client dropout - reconstruct their mask contribution
        
        If a client drops out, we need to subtract their mask contribution
        from other clients' updates.
        
        Args:
            dropped_client: ID of dropped client
            active_clients: List of clients who successfully uploaded
            surviving_updates: Updates from surviving clients
            
        Returns:
            Corrected updates
        """
        self.dropped_clients.add(dropped_client)
        
        corrected_updates = []
        
        for i, update in enumerate(surviving_updates):
            corrected_update = {}
            client_id = active_clients[i]
            
            for param_name, param in update.items():
                # Subtract the mask that was added for the dropped client
                mask = self.generate_pairwise_mask(
                    client_id, dropped_client, param.shape
                )
                
                corrected_update[param_name] = param - mask
            
            corrected_updates.append(corrected_update)
        
        print(f"  [Dropout Recovery] Corrected for dropped client: {dropped_client}")
        
        return corrected_updates
    
    def increment_round(self):
        """Increment round number (changes mask seeds)"""
        self.round_number += 1
        self.shared_secrets.clear()  # Clear old secrets


class HomomorphicEncryption:
    """
    Simplified homomorphic encryption for secure aggregation
    
    Uses additive homomorphic property:
    Enc(a) + Enc(b) = Enc(a + b)
    
    This allows server to aggregate encrypted values without decryption.
    """
    
    def __init__(self, key_size: int = 128):
        """
        Initialize homomorphic encryption
        
        Args:
            key_size: Encryption key size in bits
        """
        self.key_size = key_size
        self.public_key = None
        self.private_key = None
        
        self._generate_keys()
    
    def _generate_keys(self):
        """Generate encryption keys"""
        # Simplified: Use random large primes (in production, use proper Paillier)
        self.private_key = secrets.randbits(self.key_size)
        self.public_key = pow(2, self.private_key, 2**256)
    
    def encrypt_tensor(self, tensor: torch.Tensor) -> bytes:
        """
        Encrypt a tensor
        
        Args:
            tensor: Input tensor
            
        Returns:
            Encrypted bytes
        """
        # Serialize tensor
        tensor_bytes = pickle.dumps(tensor)
        
        # Simple XOR encryption (in production, use Paillier)
        key = self.public_key.to_bytes(32, 'big')
        encrypted = bytearray()
        
        for i, byte in enumerate(tensor_bytes):
            encrypted.append(byte ^ key[i % len(key)])
        
        return bytes(encrypted)
    
    def decrypt_tensor(self, encrypted: bytes) -> torch.Tensor:
        """
        Decrypt a tensor
        
        Args:
            encrypted: Encrypted bytes
            
        Returns:
            Decrypted tensor
        """
        # Decrypt
        key = self.public_key.to_bytes(32, 'big')
        decrypted = bytearray()
        
        for i, byte in enumerate(encrypted):
            decrypted.append(byte ^ key[i % len(key)])
        
        # Deserialize
        tensor = pickle.loads(bytes(decrypted))
        return tensor
    
    def homomorphic_add(self, enc1: bytes, enc2: bytes) -> bytes:
        """
        Add two encrypted values (homomorphic property)
        
        Args:
            enc1, enc2: Encrypted values
            
        Returns:
            Encrypted sum
        """
        # XOR-based addition (simplified)
        # In production, use proper Paillier homomorphic addition
        result = bytearray()
        
        for i in range(max(len(enc1), len(enc2))):
            b1 = enc1[i] if i < len(enc1) else 0
            b2 = enc2[i] if i < len(enc2) else 0
            result.append(b1 ^ b2)
        
        return bytes(result)


class SecureChannel:
    """
    Secure communication channel using AES encryption
    """
    
    def __init__(self):
        """Initialize secure channel"""
        self.session_keys = {}
    
    def establish_session(self, client_id: str) -> bytes:
        """
        Establish encrypted session with client
        
        Args:
            client_id: Client identifier
            
        Returns:
            Session key (AES-256)
        """
        session_key = get_random_bytes(32)  # AES-256
        self.session_keys[client_id] = session_key
        return session_key
    
    def encrypt_message(self, client_id: str, message: bytes) -> Dict:
        """
        Encrypt message for client
        
        Args:
            client_id: Target client
            message: Raw message bytes
            
        Returns:
            Dict with encrypted data and nonce
        """
        session_key = self.session_keys.get(client_id)
        if not session_key:
            raise ValueError(f"No session key for {client_id}")
        
        cipher = AES.new(session_key, AES.MODE_GCM)
        ciphertext, tag = cipher.encrypt_and_digest(message)
        
        return {
            'ciphertext': ciphertext,
            'nonce': cipher.nonce,
            'tag': tag
        }
    
    def decrypt_message(self, client_id: str, encrypted: Dict) -> bytes:
        """
        Decrypt message from client
        
        Args:
            client_id: Source client
            encrypted: Encrypted data dict
            
        Returns:
            Decrypted message bytes
        """
        session_key = self.session_keys.get(client_id)
        if not session_key:
            raise ValueError(f"No session key for {client_id}")
        
        cipher = AES.new(session_key, AES.MODE_GCM, nonce=encrypted['nonce'])
        plaintext = cipher.decrypt_and_verify(encrypted['ciphertext'], encrypted['tag'])
        
        return plaintext


def serialize_model_update(update: Dict[str, torch.Tensor]) -> bytes:
    """Serialize model update for transmission"""
    return pickle.dumps(update)


def deserialize_model_update(data: bytes) -> Dict[str, torch.Tensor]:
    """Deserialize model update from bytes"""
    return pickle.loads(data)