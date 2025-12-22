"""
Byzantine Defense Mechanisms for Federated Learning
Implements multiple defense strategies against malicious clients
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import warnings

class ByzantineDefense:
    """
    Byzantine-robust aggregation methods
    
    Supported Methods:
    - Krum: Select most representative update
    - Median: Coordinate-wise median
    - Trimmed Mean: Remove outliers, then average
    - Norm Filtering: Filter updates with abnormal norms
    """
    
    def __init__(self, method='krum', detection_threshold=2.5):
        """
        Initialize Byzantine defense
        
        Args:
            method: Defense method ('krum', 'median', 'trimmed_mean', 'norm_filter')
            detection_threshold: Threshold for anomaly detection (in standard deviations)
        """
        self.method = method
        self.detection_threshold = detection_threshold
        self.suspicious_clients = defaultdict(int)  # Track suspicious behavior
        self.client_history = defaultdict(list)  # Track client update history
        
    def aggregate_updates(self, client_updates: List[Dict], 
                         client_weights: List[float]) -> Tuple[Dict, List[str]]:
        """
        Aggregate client updates with Byzantine defense
        
        Args:
            client_updates: List of client model updates
            client_weights: List of client weights (e.g., num_samples)
            
        Returns:
            Tuple of (aggregated_update, list_of_rejected_clients)
        """
        if not client_updates:
            raise ValueError("No client updates provided")
        
        client_ids = [u.get('client_id', f'client_{i}') for i, u in enumerate(client_updates)]
        
        # Extract model updates
        model_updates = [u['generator_updates'] for u in client_updates]
        
        # Detect Byzantine clients
        rejected_clients = []
        
        if self.method == 'krum':
            aggregated, rejected = self._krum_aggregation(
                model_updates, client_ids, client_weights
            )
        elif self.method == 'median':
            aggregated, rejected = self._median_aggregation(
                model_updates, client_ids
            )
        elif self.method == 'trimmed_mean':
            aggregated, rejected = self._trimmed_mean_aggregation(
                model_updates, client_ids, client_weights
            )
        elif self.method == 'norm_filter':
            aggregated, rejected = self._norm_filter_aggregation(
                model_updates, client_ids, client_weights
            )
        else:
            # Fallback to standard weighted average
            aggregated, rejected = self._weighted_average(
                model_updates, client_ids, client_weights
            )
        
        # Update suspicious client tracker
        for client_id in rejected:
            self.suspicious_clients[client_id] += 1
        
        return aggregated, rejected
    
    def _krum_aggregation(self, updates: List[Dict], client_ids: List[str],
                         weights: List[float]) -> Tuple[Dict, List[str]]:
        """
        Krum: Select update closest to others (most representative)
        Paper: "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
        """
        n = len(updates)
        f = max(1, n // 4)  # Assume up to 25% Byzantine clients
        
        # Flatten updates to vectors
        vectors = []
        for update in updates:
            vec = torch.cat([param.flatten() for param in update.values()])
            vectors.append(vec)
        
        # Compute pairwise distances
        distances = torch.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = torch.norm(vectors[i] - vectors[j]).item()
                distances[i, j] = dist
                distances[j, i] = dist
        
        # For each update, sum distances to n-f-2 closest neighbors
        scores = []
        for i in range(n):
            dists = distances[i].sort()[0]
            score = dists[:n-f-2].sum().item()
            scores.append(score)
        
        # Select update with minimum score (most representative)
        selected_idx = np.argmin(scores)
        
        # Mark others as potentially suspicious (but don't reject all)
        rejected = []
        threshold = scores[selected_idx] * 2.0
        for i, score in enumerate(scores):
            if score > threshold and i != selected_idx:
                rejected.append(client_ids[i])
        
        print(f"  [Krum] Selected client: {client_ids[selected_idx]}")
        if rejected:
            print(f"  [Krum] Suspicious clients: {rejected}")
        
        return updates[selected_idx], rejected
    
    def _median_aggregation(self, updates: List[Dict], 
                           client_ids: List[str]) -> Tuple[Dict, List[str]]:
        """
        Coordinate-wise median: Robust to outliers
        """
        aggregated = {}
        rejected = []
        
        for key in updates[0].keys():
            # Stack all parameters
            params = torch.stack([u[key] for u in updates])
            
            # Compute coordinate-wise median
            median_params = torch.median(params, dim=0)[0]
            
            # Detect outliers (parameters far from median)
            for i, u in enumerate(updates):
                diff = torch.norm(u[key] - median_params)
                median_norm = torch.norm(median_params)
                
                if median_norm > 0 and diff / median_norm > self.detection_threshold:
                    if client_ids[i] not in rejected:
                        rejected.append(client_ids[i])
            
            aggregated[key] = median_params
        
        if rejected:
            print(f"  [Median] Rejected outlier clients: {rejected}")
        
        return aggregated, rejected
    
    def _trimmed_mean_aggregation(self, updates: List[Dict], client_ids: List[str],
                                  weights: List[float]) -> Tuple[Dict, List[str]]:
        """
        Trimmed Mean: Remove top/bottom outliers, then average
        """
        trim_ratio = 0.2  # Remove top/bottom 20%
        n_trim = max(1, int(len(updates) * trim_ratio))
        
        aggregated = {}
        rejected = []
        
        for key in updates[0].keys():
            params = torch.stack([u[key] for u in updates])
            
            # Compute norms for each client's parameters
            norms = torch.tensor([torch.norm(params[i]).item() for i in range(len(updates))])
            
            # Sort by norm
            sorted_indices = torch.argsort(norms)
            
            # Remove top and bottom n_trim
            valid_indices = sorted_indices[n_trim:-n_trim] if n_trim > 0 else sorted_indices
            
            # Track rejected clients
            rejected_indices = set(range(len(updates))) - set(valid_indices.tolist())
            for idx in rejected_indices:
                if client_ids[idx] not in rejected:
                    rejected.append(client_ids[idx])
            
            # Compute weighted mean of remaining
            valid_params = params[valid_indices]
            valid_weights = torch.tensor([weights[i] for i in valid_indices])
            valid_weights = valid_weights / valid_weights.sum()
            
            aggregated[key] = (valid_params * valid_weights.view(-1, 1, 1, 1)).sum(dim=0)
        
        if rejected:
            print(f"  [Trimmed Mean] Trimmed outlier clients: {rejected}")
        
        return aggregated, rejected
    
    def _norm_filter_aggregation(self, updates: List[Dict], client_ids: List[str],
                                 weights: List[float]) -> Tuple[Dict, List[str]]:
        """
        Norm Filtering: Reject updates with abnormal gradient norms
        """
        # Compute norm of each update
        norms = []
        for update in updates:
            total_norm = sum(torch.norm(param).item() ** 2 for param in update.values())
            norms.append(np.sqrt(total_norm))
        
        norms = np.array(norms)
        
        # Compute statistics
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        
        # Filter outliers
        valid_updates = []
        valid_weights = []
        valid_ids = []
        rejected = []
        
        for i, norm in enumerate(norms):
            z_score = abs(norm - mean_norm) / (std_norm + 1e-10)
            
            if z_score <= self.detection_threshold:
                valid_updates.append(updates[i])
                valid_weights.append(weights[i])
                valid_ids.append(client_ids[i])
            else:
                rejected.append(client_ids[i])
                print(f"  [Norm Filter] Rejected {client_ids[i]}: "
                      f"norm={norm:.4f}, z-score={z_score:.2f}")
        
        # If all rejected, use all (safety fallback)
        if not valid_updates:
            print("  [Norm Filter] WARNING: All clients rejected, using all updates")
            valid_updates = updates
            valid_weights = weights
            rejected = []
        
        # Weighted average of valid updates
        aggregated, _ = self._weighted_average(valid_updates, valid_ids, valid_weights)
        
        return aggregated, rejected
    
    def _weighted_average(self, updates: List[Dict], client_ids: List[str],
                         weights: List[float]) -> Tuple[Dict, List[str]]:
        """Standard weighted averaging (no Byzantine defense)"""
        aggregated = {}
        total_weight = sum(weights)
        
        for key in updates[0].keys():
            weighted_sum = torch.zeros_like(updates[0][key])
            
            for update, weight in zip(updates, weights):
                weighted_sum += update[key] * (weight / total_weight)
            
            aggregated[key] = weighted_sum
        
        return aggregated, []
    
    def get_client_reputation(self, client_id: str) -> float:
        """
        Get reputation score for a client (0-1, lower is worse)
        
        Returns:
            Reputation score: 1.0 = trusted, 0.0 = highly suspicious
        """
        suspicion_count = self.suspicious_clients.get(client_id, 0)
        
        # Exponential decay: rep = exp(-suspicion_count / 5)
        reputation = np.exp(-suspicion_count / 5.0)
        
        return reputation
    
    def should_exclude_client(self, client_id: str, threshold: int = 3) -> bool:
        """
        Check if client should be permanently excluded
        
        Args:
            client_id: Client identifier
            threshold: Number of suspicious rounds before exclusion
            
        Returns:
            True if client should be excluded
        """
        return self.suspicious_clients.get(client_id, 0) >= threshold
    
    def reset_client_reputation(self, client_id: str):
        """Reset reputation for a client (e.g., after verification)"""
        if client_id in self.suspicious_clients:
            self.suspicious_clients[client_id] = 0
    
    def get_defense_stats(self) -> Dict:
        """Get statistics about Byzantine defense"""
        return {
            'method': self.method,
            'detection_threshold': self.detection_threshold,
            'suspicious_clients': dict(self.suspicious_clients),
            'total_suspicious_events': sum(self.suspicious_clients.values())
        }

