"""
Comprehensive Metrics Logger for Research Paper
Tracks training metrics, saves to JSON, generates plots
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


class MetricsLogger:
    """
    Advanced metrics logger for federated learning research
    Tracks: loss, accuracy, communication cost, privacy budget, Byzantine events
    """
    
    def __init__(self, save_dir: str = "metrics"):
        """
        Initialize metrics logger
        
        Args:
            save_dir: Directory to save metrics and plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Main metrics storage
        self.metrics = {
            'training_rounds': [],          # Per-round metrics
            'client_metrics': defaultdict(list),  # Per-client metrics
            'global_metrics': [],           # Global model metrics
            'byzantine_events': [],         # Byzantine defense events
            'communication_cost': [],       # Network traffic
            'privacy_budget': [],           # Differential privacy tracking
            'metadata': {
                'start_time': datetime.now().isoformat(),
                'experiment_name': 'federated_rag',
                'num_clients': 0
            }
        }
        
        print(f"✓ MetricsLogger initialized. Saving to: {save_dir}")
    
    def log_training_round(self, round_num: int, round_data: Dict):
        """
        Log complete training round data
        
        Args:
            round_num: Training round number
            round_data: Dict with round information from server
        """
        timestamp = datetime.now().isoformat()
        
        # Extract key metrics
        round_metrics = {
            'round': round_num,
            'timestamp': timestamp,
            'avg_loss': round_data.get('avg_loss', None),
            'num_clients': round_data.get('num_clients', 0),
            'total_samples': round_data.get('total_samples', 0),
            'rejected_clients': round_data.get('rejected_clients', []),
            'num_rejected': len(round_data.get('rejected_clients', [])),
            'aggregation_method': round_data.get('aggregation_method', 'fedavg')
        }
        
        self.metrics['training_rounds'].append(round_metrics)
        
        print(f"📊 Round {round_num} logged: Loss={round_metrics['avg_loss']:.4f}, "
              f"Clients={round_metrics['num_clients']}, Rejected={round_metrics['num_rejected']}")
    
    def log_client_training(self, round_num: int, client_id: str, client_data: Dict):
        """
        Log individual client training data
        
        Args:
            round_num: Training round number
            client_id: Client identifier
            client_data: Dict with client training results
        """
        timestamp = datetime.now().isoformat()
        
        client_metrics = {
            'round': round_num,
            'timestamp': timestamp,
            'client_id': client_id,
            'loss': client_data.get('loss', None),
            'num_samples': client_data.get('num_samples', 0),
            'epoch_losses': client_data.get('epoch_losses', []),
            'status': client_data.get('status', 'unknown')
        }
        
        self.metrics['client_metrics'][client_id].append(client_metrics)
        
        print(f"  📝 {client_id}: Loss={client_metrics['loss']:.4f}, "
              f"Samples={client_metrics['num_samples']}")
    
    def log_byzantine_event(self, round_num: int, event_data: Dict):
        """
        Log Byzantine defense events
        
        Args:
            round_num: Training round number
            event_data: Dict with Byzantine event information
        """
        event = {
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'rejected_clients': event_data.get('rejected_clients', []),
            'defense_method': event_data.get('method', 'unknown'),
            'suspicious_count': len(event_data.get('rejected_clients', []))
        }
        
        self.metrics['byzantine_events'].append(event)
        
        if event['suspicious_count'] > 0:
            print(f"  ⚠️  Byzantine: Rejected {event['suspicious_count']} clients")
    
    def log_communication_cost(self, round_num: int, bytes_sent: int, bytes_received: int):
        """
        Log communication cost for the round
        
        Args:
            round_num: Training round number
            bytes_sent: Bytes sent to server
            bytes_received: Bytes received from server
        """
        comm_data = {
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'bytes_sent': bytes_sent,
            'bytes_received': bytes_received,
            'total_bytes': bytes_sent + bytes_received,
            'mb_sent': bytes_sent / (1024 * 1024),
            'mb_received': bytes_received / (1024 * 1024),
            'total_mb': (bytes_sent + bytes_received) / (1024 * 1024)
        }
        
        self.metrics['communication_cost'].append(comm_data)
        
        print(f"  📡 Communication: {comm_data['total_mb']:.2f} MB")
    
    def log_privacy_budget(self, round_num: int, epsilon: float, delta: float = 1e-5,
                          noise_multiplier: float = 0.1):
        """
        Log differential privacy budget consumption
        
        Args:
            round_num: Training round number
            epsilon: Privacy epsilon for this round
            delta: Privacy delta
            noise_multiplier: DP noise multiplier used
        """
        privacy_data = {
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'epsilon': epsilon,
            'delta': delta,
            'noise_multiplier': noise_multiplier,
            'cumulative_epsilon': self._calculate_cumulative_epsilon()
        }
        
        self.metrics['privacy_budget'].append(privacy_data)
        
        print(f"  🔒 Privacy: ε={epsilon:.2f}, Cumulative ε={privacy_data['cumulative_epsilon']:.2f}")
    
    def _calculate_cumulative_epsilon(self) -> float:
        """Calculate cumulative privacy budget using composition"""
        if not self.metrics['privacy_budget']:
            return 0.0
        
        # Simple composition (for more accuracy, use advanced composition)
        total_epsilon = sum(p['epsilon'] for p in self.metrics['privacy_budget'])
        return total_epsilon  # This should now work correctly
    
    def save_metrics(self, filename: str = "training_metrics.json"):
        """
        Save all metrics to JSON file
        
        Args:
            filename: Output filename
        """
        filepath = os.path.join(self.save_dir, filename)
        
        # Add summary statistics
        self.metrics['summary'] = self._generate_summary()
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"✓ Metrics saved to: {filepath}")
        return filepath
    
    def _generate_summary(self) -> Dict:
        """Generate summary statistics for the paper"""
        summary = {
            'total_rounds': len(self.metrics['training_rounds']),
            'total_clients': len(self.metrics['client_metrics']),
            'total_byzantine_events': len(self.metrics['byzantine_events']),
            'final_loss': None,
            'avg_loss_improvement': None,
            'total_communication_mb': 0,
            'final_privacy_budget': 0
        }
        
        # Calculate loss improvement
        if self.metrics['training_rounds']:
            losses = [r['avg_loss'] for r in self.metrics['training_rounds'] if r['avg_loss']]
            if losses:
                summary['initial_loss'] = losses[0]
                summary['final_loss'] = losses[-1]
                summary['avg_loss_improvement'] = losses[0] - losses[-1]
                summary['loss_reduction_percent'] = (
                    (losses[0] - losses[-1]) / losses[0] * 100 if losses[0] > 0 else 0
                )
        
        # Calculate total communication
        if self.metrics['communication_cost']:
            summary['total_communication_mb'] = sum(
                c['total_mb'] for c in self.metrics['communication_cost']
            )
        
        # Calculate final privacy budget
        if self.metrics['privacy_budget']:
            summary['final_privacy_budget'] = self.metrics['privacy_budget'][-1]['cumulative_epsilon']
        
        return summary
    
    def plot_training_curves(self, save_path: str = None):
        """
        Generate training curves plot for paper
        
        Args:
            save_path: Path to save figure (default: metrics/training_curves.png)
        """
        if not self.metrics['training_rounds']:
            print("⚠️  No training data to plot")
            return
        
        save_path = save_path or os.path.join(self.save_dir, "training_curves.png")
        
        # Extract data
        rounds = [r['round'] for r in self.metrics['training_rounds']]
        losses = [r['avg_loss'] for r in self.metrics['training_rounds'] if r['avg_loss']]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Federated Learning Training Metrics', fontsize=16, fontweight='bold')
        
        # Plot 1: Loss curve
        axes[0, 0].plot(rounds[:len(losses)], losses, marker='o', linewidth=2, markersize=6)
        axes[0, 0].set_xlabel('Training Round', fontsize=12)
        axes[0, 0].set_ylabel('Average Loss', fontsize=12)
        axes[0, 0].set_title('Training Loss Convergence', fontsize=13, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Per-client losses
        for client_id, client_data in self.metrics['client_metrics'].items():
            client_rounds = [d['round'] for d in client_data]
            client_losses = [d['loss'] for d in client_data if d['loss']]
            axes[0, 1].plot(client_rounds[:len(client_losses)], client_losses, 
                          marker='s', label=client_id, linewidth=2, markersize=4)
        
        axes[0, 1].set_xlabel('Training Round', fontsize=12)
        axes[0, 1].set_ylabel('Client Loss', fontsize=12)
        axes[0, 1].set_title('Per-Client Loss Curves', fontsize=13, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Communication cost
        if self.metrics['communication_cost']:
            comm_rounds = [c['round'] for c in self.metrics['communication_cost']]
            comm_mb = [c['total_mb'] for c in self.metrics['communication_cost']]
            axes[1, 0].bar(comm_rounds, comm_mb, color='steelblue', alpha=0.7)
            axes[1, 0].set_xlabel('Training Round', fontsize=12)
            axes[1, 0].set_ylabel('Communication (MB)', fontsize=12)
            axes[1, 0].set_title('Communication Cost per Round', fontsize=13, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Privacy budget
        if self.metrics['privacy_budget']:
            priv_rounds = [p['round'] for p in self.metrics['privacy_budget']]
            cumulative_eps = [p['cumulative_epsilon'] for p in self.metrics['privacy_budget']]
            axes[1, 1].plot(priv_rounds, cumulative_eps, marker='d', 
                          color='red', linewidth=2, markersize=6)
            axes[1, 1].set_xlabel('Training Round', fontsize=12)
            axes[1, 1].set_ylabel('Cumulative ε', fontsize=12)
            axes[1, 1].set_title('Privacy Budget Consumption', fontsize=13, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training curves saved to: {save_path}")
        
        return save_path
    
    def plot_comparison_bar(self, comparison_data: Dict, save_path: str = None):
        """
        Generate comparison bar chart for baseline methods
        
        Args:
            comparison_data: Dict mapping method names to metrics
            save_path: Path to save figure
        """
        save_path = save_path or os.path.join(self.save_dir, "baseline_comparison.png")
        
        methods = list(comparison_data.keys())
        losses = [comparison_data[m]['loss'] for m in methods]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(methods, losses, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
        
        ax.set_ylabel('Final Loss', fontsize=12)
        ax.set_title('Comparison with Baseline Methods', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved to: {save_path}")
        
        return save_path
    
    def generate_latex_table(self, save_path: str = None) -> str:
        """
        Generate LaTeX table for paper
        
        Args:
            save_path: Path to save .tex file
        
        Returns:
            LaTeX table string
        """
        save_path = save_path or os.path.join(self.save_dir, "results_table.tex")
        
        summary = self._generate_summary()
        
        latex_table = r"""
\begin{table}[htbp]
\centering
\caption{Federated Learning Training Results}
\label{tab:results}
\begin{tabular}{lc}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Total Training Rounds & """ + str(summary['total_rounds']) + r""" \\
Number of Clients & """ + str(summary['total_clients']) + r""" \\
Initial Loss & """ + f"{summary.get('initial_loss', 0):.4f}" + r""" \\
Final Loss & """ + f"{summary.get('final_loss', 0):.4f}" + r""" \\
Loss Improvement & """ + f"{summary.get('avg_loss_improvement', 0):.4f}" + r""" \\
Loss Reduction (\%) & """ + f"{summary.get('loss_reduction_percent', 0):.2f}\%" + r""" \\
Total Communication (MB) & """ + f"{summary['total_communication_mb']:.2f}" + r""" \\
Final Privacy Budget ($\epsilon$) & """ + f"{summary['final_privacy_budget']:.2f}" + r""" \\
Byzantine Events & """ + str(summary['total_byzantine_events']) + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
        
        with open(save_path, 'w') as f:
            f.write(latex_table)
        
        print(f"✓ LaTeX table saved to: {save_path}")
        return latex_table
    
    def print_summary(self):
        """Print summary statistics to console"""
        summary = self._generate_summary()
        
        print("\n" + "="*60)
        print("📊 TRAINING SUMMARY")
        print("="*60)
        print(f"Total Rounds:          {summary['total_rounds']}")
        print(f"Total Clients:         {summary['total_clients']}")
        print(f"Initial Loss:          {summary.get('initial_loss', 'N/A'):.4f}" if summary.get('initial_loss') else "Initial Loss:          N/A")
        print(f"Final Loss:            {summary.get('final_loss', 'N/A'):.4f}" if summary.get('final_loss') else "Final Loss:            N/A")
        print(f"Loss Improvement:      {summary.get('avg_loss_improvement', 'N/A'):.4f}" if summary.get('avg_loss_improvement') else "Loss Improvement:      N/A")
        print(f"Loss Reduction:        {summary.get('loss_reduction_percent', 'N/A'):.2f}%" if summary.get('loss_reduction_percent') else "Loss Reduction:        N/A")
        print(f"Communication (MB):    {summary['total_communication_mb']:.2f}")
        print(f"Privacy Budget (ε):    {summary['final_privacy_budget']:.2f}")
        print(f"Byzantine Events:      {summary['total_byzantine_events']}")
        print("="*60 + "\n")


# Utility function to estimate model size
def estimate_model_size(state_dict: Dict) -> int:
    """
    Estimate size of model update in bytes
    
    Args:
        state_dict: Model state dictionary
    
    Returns:
        Approximate size in bytes
    """
    import sys
    total_bytes = 0
    
    for key, value in state_dict.items():
        # Each float32 parameter is 4 bytes
        if hasattr(value, 'numel'):
            total_bytes += value.numel() * 4
        else:
            total_bytes += sys.getsizeof(value)
    
    return total_bytes