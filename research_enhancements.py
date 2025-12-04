"""
Research-Grade Enhancements for Federated RAG System
Add these features to make your paper Level 3 (Top Tier)
"""
import time
import numpy as np
from typing import List, Dict
import json

class ResearchMetrics:
    """Track advanced metrics for research paper"""
    
    def __init__(self):
        self.metrics = {
            'convergence_time': [],
            'communication_cost': [],
            'privacy_budget': [],
            'answer_quality': [],
            'retrieval_accuracy': []
        }
    
    def log_round_metrics(self, round_num, loss, num_updates, privacy_epsilon):
        """Log metrics for each training round"""
        self.metrics['convergence_time'].append({
            'round': round_num,
            'loss': loss,
            'timestamp': time.time()
        })
        self.metrics['communication_cost'].append({
            'round': round_num,
            'num_updates': num_updates,
            'bytes_transferred': num_updates * 1024  # Estimate
        })
        self.metrics['privacy_budget'].append({
            'round': round_num,
            'epsilon': privacy_epsilon
        })
    
    def calculate_convergence_rate(self):
        """Calculate convergence rate for paper"""
        losses = [m['loss'] for m in self.metrics['convergence_time']]
        if len(losses) < 2:
            return 0
        
        # Calculate rate of loss decrease
        rate = (losses[0] - losses[-1]) / len(losses)
        return rate
    
    def get_communication_efficiency(self):
        """Communication cost per accuracy improvement"""
        total_bytes = sum(m['bytes_transferred'] for m in self.metrics['communication_cost'])
        losses = [m['loss'] for m in self.metrics['convergence_time']]
        
        if len(losses) < 2:
            return 0
        
        accuracy_improvement = losses[0] - losses[-1]
        return total_bytes / accuracy_improvement if accuracy_improvement > 0 else float('inf')
    
    def export_for_paper(self, filename='research_metrics.json'):
        """Export all metrics for paper figures"""
        export_data = {
            'convergence': self.metrics['convergence_time'],
            'communication': self.metrics['communication_cost'],
            'privacy': self.metrics['privacy_budget'],
            'summary': {
                'convergence_rate': self.calculate_convergence_rate(),
                'communication_efficiency': self.get_communication_efficiency(),
                'total_rounds': len(self.metrics['convergence_time'])
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return export_data


class AdaptivePrivacyBudget:
    """Novel contribution: Adaptive privacy budget allocation"""
    
    def __init__(self, initial_epsilon=1.0, decay_rate=0.9):
        self.epsilon = initial_epsilon
        self.decay_rate = decay_rate
        self.round = 0
    
    def get_current_epsilon(self, loss):
        """Adapt privacy based on training progress"""
        # Novel idea: Reduce privacy budget as model converges
        self.round += 1
        
        # If loss is high, use more privacy (less noise)
        # If loss is low, use less privacy (more noise) - model already good
        if loss > 2.0:
            return self.epsilon
        elif loss > 1.0:
            return self.epsilon * 0.7
        else:
            return self.epsilon * 0.5
    
    def update_epsilon(self):
        """Decay epsilon over time"""
        self.epsilon *= self.decay_rate


class HeterogeneousDataSimulator:
    """Simulate heterogeneous data distribution across clients"""
    
    @staticmethod
    def create_non_iid_split(documents, num_clients=5, alpha=0.5):
        """
        Create non-IID data split using Dirichlet distribution
        Important for research: Shows system works with realistic data
        
        Args:
            documents: List of documents
            num_clients: Number of clients
            alpha: Dirichlet parameter (lower = more heterogeneous)
        
        Returns:
            Dict mapping client_id to document indices
        """
        num_docs = len(documents)
        
        # Dirichlet distribution for non-IID split
        proportions = np.random.dirichlet([alpha] * num_clients, 1)[0]
        
        # Assign documents to clients based on proportions
        client_data = {}
        start_idx = 0
        
        for i in range(num_clients):
            num_docs_for_client = int(proportions[i] * num_docs)
            end_idx = start_idx + num_docs_for_client
            
            if i == num_clients - 1:  # Last client gets remaining
                end_idx = num_docs
            
            client_data[f'client_{i}'] = list(range(start_idx, end_idx))
            start_idx = end_idx
        
        return client_data


class BaselineComparison:
    """Compare with baseline federated methods"""
    
    METHODS = {
        'fedavg': 'Standard FedAvg',
        'fedprox': 'FedProx with proximal term',
        'fedopt': 'FedOpt with adaptive optimization',
        'ours': 'Our Method (Adaptive Privacy + LoRA)'
    }
    
    def __init__(self):
        self.results = {method: [] for method in self.METHODS.keys()}
    
    def log_method_performance(self, method, round_num, loss, accuracy):
        """Log performance for comparison"""
        self.results[method].append({
            'round': round_num,
            'loss': loss,
            'accuracy': accuracy
        })
    
    def generate_comparison_table(self):
        """Generate comparison table for paper"""
        table = []
        
        for method, name in self.METHODS.items():
            if self.results[method]:
                final_loss = self.results[method][-1]['loss']
                final_accuracy = self.results[method][-1]['accuracy']
                rounds = len(self.results[method])
                
                table.append({
                    'Method': name,
                    'Final Loss': f"{final_loss:.4f}",
                    'Accuracy': f"{final_accuracy:.2f}%",
                    'Rounds': rounds
                })
        
        return table


class AblationStudy:
    """Run ablation studies for paper"""
    
    def __init__(self):
        self.configurations = {
            'full': 'Full model (LoRA + DP + Adaptive)',
            'no_lora': 'Without LoRA',
            'no_dp': 'Without Differential Privacy',
            'no_adaptive': 'Without Adaptive Privacy',
            'baseline': 'Standard FedAvg'
        }
        self.results = {config: [] for config in self.configurations.keys()}
    
    def log_ablation_result(self, config, metric_name, value):
        """Log ablation study results"""
        self.results[config].append({
            'metric': metric_name,
            'value': value
        })
    
    def generate_ablation_table(self):
        """Generate ablation table showing contribution of each component"""
        table = []
        
        for config, description in self.configurations.items():
            if self.results[config]:
                accuracy = next((r['value'] for r in self.results[config] 
                               if r['metric'] == 'accuracy'), 0)
                privacy = next((r['value'] for r in self.results[config] 
                              if r['metric'] == 'privacy'), 0)
                speed = next((r['value'] for r in self.results[config] 
                            if r['metric'] == 'speed'), 0)
                
                table.append({
                    'Configuration': description,
                    'Accuracy': f"{accuracy:.2f}%",
                    'Privacy (ε)': f"{privacy:.2f}",
                    'Speed (s)': f"{speed:.1f}"
                })
        
        return table


class QualityMetrics:
    """Evaluate answer quality for paper"""
    
    @staticmethod
    def calculate_bleu(reference, hypothesis):
        """Simple BLEU-1 score"""
        ref_tokens = set(reference.lower().split())
        hyp_tokens = hypothesis.lower().split()
        
        if not hyp_tokens:
            return 0.0
        
        matches = sum(1 for token in hyp_tokens if token in ref_tokens)
        return matches / len(hyp_tokens)
    
    @staticmethod
    def calculate_rouge_l(reference, hypothesis):
        """Simple ROUGE-L score"""
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        if not ref_tokens or not hyp_tokens:
            return 0.0
        
        # Find longest common subsequence
        m, n = len(ref_tokens), len(hyp_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i-1] == hyp_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        precision = lcs_length / n if n > 0 else 0
        recall = lcs_length / m if m > 0 else 0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    @staticmethod
    def evaluate_answer_quality(reference_answers, generated_answers):
        """Evaluate all answers and return average metrics"""
        bleu_scores = []
        rouge_scores = []
        
        for ref, gen in zip(reference_answers, generated_answers):
            bleu_scores.append(QualityMetrics.calculate_bleu(ref, gen))
            rouge_scores.append(QualityMetrics.calculate_rouge_l(ref, gen))
        
        return {
            'avg_bleu': np.mean(bleu_scores),
            'avg_rouge': np.mean(rouge_scores),
            'std_bleu': np.std(bleu_scores),
            'std_rouge': np.std(rouge_scores)
        }


# Example usage for paper experiments
def run_full_experiment():
    """Complete experiment pipeline for paper"""
    
    print("Running Research-Grade Experiments...")
    print("=" * 60)
    
    # 1. Initialize metrics tracker
    metrics = ResearchMetrics()
    
    # 2. Run experiments with different configurations
    ablation = AblationStudy()
    baseline = BaselineComparison()
    
    # 3. Simulate multiple rounds
    for round_num in range(1, 11):
        # Simulate training
        loss = 3.0 - (round_num * 0.2)  # Decreasing loss
        accuracy = 50 + (round_num * 3)  # Increasing accuracy
        
        # Log metrics
        metrics.log_round_metrics(round_num, loss, num_updates=2, privacy_epsilon=1.0)
        baseline.log_method_performance('ours', round_num, loss, accuracy)
        
        print(f"Round {round_num}: Loss={loss:.4f}, Accuracy={accuracy:.1f}%")
    
    # 4. Export results
    results = metrics.export_for_paper()
    print("\n" + "=" * 60)
    print("Results exported to research_metrics.json")
    print(f"Convergence Rate: {results['summary']['convergence_rate']:.4f}")
    print(f"Communication Efficiency: {results['summary']['communication_efficiency']:.2f} bytes/loss")
    
    # 5. Generate comparison table
    print("\n" + "=" * 60)
    print("Baseline Comparison (Table for Paper):")
    comparison = baseline.generate_comparison_table()
    for row in comparison:
        print(f"  {row}")
    
    return results


if __name__ == "__main__":
    run_full_experiment()