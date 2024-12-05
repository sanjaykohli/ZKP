import hashlib
import time
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
import random
import math
from collections import defaultdict
import statistics
from concurrent.futures import ThreadPoolExecutor
import numpy as np

@dataclass
class Transaction:
    sender: str
    receiver: str
    amount: float
    timestamp: float
    proof: Optional[bytes] = None
    transaction_id: str = field(init=False)
    
    def __post_init__(self):
        # Generate unique transaction ID using sender, receiver, amount, and timestamp
        self.transaction_id = hashlib.sha256(
            f"{self.sender}{self.receiver}{self.amount}{self.timestamp}".encode()
        ).hexdigest()[:16]

class BulletProof:
    """Optimized implementation of Bulletproofs ZKP protocol simulation"""
    def __init__(self, secret: int, security_level: int = 1000):
        self.secret = secret
        self.security_level = security_level
        self._cache = {}
    
    def generate_proof(self) -> bytes:
        cache_key = f"{self.secret}_{self.security_level}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Use numpy for faster array operations
        random_values = np.random.bytes(32)
        proof = hashlib.blake2b(
            str(self.secret).encode() + random_values,
            digest_size=32
        ).digest()
        
        self._cache[cache_key] = proof
        return proof
    
    def verify_proof(self, proof: bytes) -> Tuple[bool, float]:
        start = time.perf_counter()
        # Use blake2b for faster hashing
        verification = hashlib.blake2b(proof, digest_size=32).digest()
        verification_time = time.perf_counter() - start
        return True, verification_time

class PrivacyEnhancedWallet:
    def __init__(self, max_workers: int = 4):
        self.transactions: List[Transaction] = []
        self.anonymity_set: Set[str] = set()
        self.node_participation: Dict[str, int] = defaultdict(int)
        self.performance_metrics = {
            'memory_usage': [],
            'cpu_usage': [],
            'bandwidth_usage': [],
            'proof_generation_times': [],
            'verification_times': []
        }
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.transaction_graph = defaultdict(set)

    def _calculate_linkability_resistance(self) -> float:
        """Calculate resistance to transaction linking attempts using graph theory metrics"""
        if not self.transactions:
            return 0.0
        
        # Calculate based on transaction graph complexity
        unique_participants = len(self.anonymity_set)
        total_edges = sum(len(connections) for connections in self.transaction_graph.values())
        
        if unique_participants <= 1:
            return 0.0
            
        # Calculate graph density as a measure of linkability resistance
        max_possible_edges = unique_participants * (unique_participants - 1)
        graph_density = total_edges / max_possible_edges if max_possible_edges > 0 else 0
        
        # Combine with entropy for better measure
        entropy = self._calculate_transaction_entropy()
        
        # Normalize to 0-1 range
        resistance = (graph_density + entropy) / (1 + math.log2(unique_participants))
        return min(1.0, resistance)
    
    def _calculate_node_participation(self) -> float:
        """Calculate the rate of node participation in verification"""
        if not self.node_participation:
            return 0.0
            
        total_possible_nodes = max(100, len(self.node_participation))
        active_nodes = len([v for v in self.node_participation.values() if v > 0])
        return active_nodes / total_possible_nodes
    
    def _calculate_consensus_confidence(self) -> float:
        """Calculate confidence in consensus decisions"""
        if not self.node_participation:
            return 0.0
        
        # Calculate based on node agreement distribution
        participation_values = list(self.node_participation.values())
        if not participation_values:
            return 0.0
            
        mean_participation = statistics.mean(participation_values)
        agreed_nodes = sum(1 for v in participation_values if v > mean_participation * 0.8)
        
        return agreed_nodes / len(participation_values)
    
    def create_transaction(self, sender: str, receiver: str, amount: float) -> Transaction:
        start_cpu = time.perf_counter()
        
        # Generate proof using thread pool
        secret = random.getrandbits(256)
        bulletproof = BulletProof(secret)
        future = self.executor.submit(bulletproof.generate_proof)
        proof = future.result()
        
        tx = Transaction(
            sender=sender,
            receiver=receiver,
            amount=amount,
            timestamp=time.time(),
            proof=proof
        )
        
        # Update transaction graph for better analysis
        self.transaction_graph[sender].add(tx.transaction_id)
        self.transaction_graph[receiver].add(tx.transaction_id)
        
        # Update metrics
        self.transactions.append(tx)
        self.anonymity_set.add(sender)
        self.anonymity_set.add(receiver)
        
        # Record performance metrics
        self._update_performance_metrics(tx, start_cpu)
        
        return tx
    
    def _update_performance_metrics(self, tx: Transaction, start_cpu: float):
        """Update all performance metrics in one pass"""
        cpu_time = time.perf_counter() - start_cpu
        self.performance_metrics['cpu_usage'].append(cpu_time)
        
        # Estimate memory usage more accurately
        tx_size = len(tx.__dict__.__str__().encode())
        self.performance_metrics['memory_usage'].append(tx_size / 1024)  # KB
        self.performance_metrics['bandwidth_usage'].append(tx_size / 1024)  # KB
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive privacy and efficiency metrics"""
        metrics = {
            # Privacy Metrics
            'anonymity_set_size': len(self.anonymity_set),
            'linkability_resistance': self._calculate_linkability_resistance(),
            'transaction_entropy': self._calculate_transaction_entropy(),
            'mixing_factor': self._calculate_mixing_factor(),
            
            # Performance Metrics
            'avg_cpu_usage_ms': statistics.mean(self.performance_metrics['cpu_usage']) * 1000,
            'avg_memory_usage_kb': statistics.mean(self.performance_metrics['memory_usage']),
            'avg_bandwidth_usage_kb': statistics.mean(self.performance_metrics['bandwidth_usage']),
            'cpu_usage_std_dev': statistics.stdev(self.performance_metrics['cpu_usage']) if len(self.performance_metrics['cpu_usage']) > 1 else 0,
            
            # Network Metrics
            'node_participation_rate': self._calculate_node_participation(),
            'consensus_confidence': self._calculate_consensus_confidence(),
            'network_resilience': self._calculate_network_resilience(),
            
            # Transaction Metrics
            'total_transactions': len(self.transactions),
            'transactions_per_second': self._calculate_tps(),
        }
        
        # Add statistical summaries
        metrics.update(self._calculate_statistical_metrics())
        
        return metrics
    
    def _calculate_transaction_entropy(self) -> float:
        """Calculate transaction entropy using Shannon entropy"""
        if not self.transactions:
            return 0.0
            
        # Calculate probability distribution of transactions per user
        tx_counts = defaultdict(int)
        for tx in self.transactions:
            tx_counts[tx.sender] += 1
            tx_counts[tx.receiver] += 1
            
        total_tx = sum(tx_counts.values())
        probabilities = [count/total_tx for count in tx_counts.values()]
        
        # Calculate Shannon entropy
        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
        return entropy
    
    def _calculate_mixing_factor(self) -> float:
        """Calculate transaction mixing factor"""
        if not self.transaction_graph:
            return 0.0
        
        # Calculate average number of connections per user
        connections_per_user = [len(connections) for connections in self.transaction_graph.values()]
        return statistics.mean(connections_per_user) if connections_per_user else 0
    
    def _calculate_network_resilience(self) -> float:
        """Calculate network resilience based on node distribution"""
        if not self.node_participation:
            return 0.0
            
        # Calculate Gini coefficient of node participation
        values = sorted(self.node_participation.values())
        n = len(values)
        index = np.arange(1, n + 1)
        return (np.sum((2 * index - n - 1) * values)) / (n * np.sum(values))
    
    def _calculate_statistical_metrics(self) -> Dict[str, float]:
        """Calculate statistical metrics for performance analysis"""
        return {
            'transaction_value_mean': statistics.mean(tx.amount for tx in self.transactions),
            'transaction_value_median': statistics.median(tx.amount for tx in self.transactions),
            'transaction_value_std_dev': statistics.stdev(tx.amount for tx in self.transactions) if len(self.transactions) > 1 else 0,
        }
    
    def _calculate_tps(self) -> float:
        """Calculate transactions per second"""
        if len(self.transactions) < 2:
            return 0.0
        
        time_diff = self.transactions[-1].timestamp - self.transactions[0].timestamp
        return len(self.transactions) / time_diff if time_diff > 0 else 0

def run_simulation(num_transactions: int = 100, num_users: int = 20, num_nodes: int = 50) -> Dict[str, float]:
    """Run enhanced simulation with more realistic network conditions"""
    wallet = PrivacyEnhancedWallet()
    
    # Use numpy for faster random number generation
    senders = np.random.randint(1, num_users + 1, num_transactions)
    receivers = np.random.randint(1, num_users + 1, num_transactions)
    amounts = np.random.uniform(0.1, 10.0, num_transactions)
    
    # Simulate transactions in parallel
    with ThreadPoolExecutor() as executor:
        futures = []
        for sender, receiver, amount in zip(senders, receivers, amounts):
            futures.append(
                executor.submit(
                    wallet.create_transaction,
                    f"user_{sender}",
                    f"user_{receiver}",
                    amount
                )
            )
        
        # Wait for all transactions to complete
        for future in futures:
            future.result()
    
    # Simulate node participation
    node_participations = np.random.randint(5, 15, num_transactions)
    node_ids = np.random.randint(1, num_nodes + 1, np.sum(node_participations))
    
    for node_id in node_ids:
        wallet.node_participation[f"node_{node_id}"] += 1
    
    return wallet.calculate_metrics()

# Run simulation with increased scale
metrics = run_simulation(num_transactions=1000, num_users=50, num_nodes=100)

# Pretty print metrics with formatting
def print_metrics(metrics: Dict[str, float]):
    print("\nPrivacy-Enhanced Wallet Metrics:")
    print("-" * 50)
    
    categories = {
        'Privacy Metrics': ['anonymity_set_size', 'linkability_resistance', 'transaction_entropy', 'mixing_factor'],
        'Performance Metrics': ['avg_cpu_usage_ms', 'avg_memory_usage_kb', 'avg_bandwidth_usage_kb', 'cpu_usage_std_dev'],
        'Network Metrics': ['node_participation_rate', 'consensus_confidence', 'network_resilience'],
        'Transaction Metrics': ['total_transactions', 'transactions_per_second', 'transaction_value_mean', 
                              'transaction_value_median', 'transaction_value_std_dev']
    }
    
    for category, metric_names in categories.items():
        print(f"\n{category}:")
        for name in metric_names:
            if name in metrics:
                value = metrics[name]
                print(f"{name.replace('_', ' ').title():.<40} {value:.4f}")

print_metrics(metrics)
