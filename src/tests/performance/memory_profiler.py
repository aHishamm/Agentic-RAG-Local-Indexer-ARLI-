import os
import psutil
import time
from memory_profiler import profile
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from indexer.utils import init_embedding_model, generate_filename_embedding, search_similar_files
from indexer.models import Indexer
from django.conf import settings

class MemoryProfiler:
    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = None

    def get_memory_usage(self):
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024

    def start_monitoring(self):
        """Start monitoring memory usage"""
        self.baseline_memory = self.get_memory_usage()
        return self.baseline_memory

    def get_memory_increase(self):
        """Get memory increase from baseline in MB"""
        if self.baseline_memory is None:
            return 0
        current = self.get_memory_usage()
        return current - self.baseline_memory

@profile
def test_model_loading():
    """Profile memory usage during model loading"""
    profiler = MemoryProfiler()
    print(f"Baseline Memory Usage: {profiler.start_monitoring():.2f} MB")

    # Test UAE-Large-V1 model loading
    print("Loading UAE-Large-V1 model...")
    start_time = time.time()
    model = init_embedding_model()
    load_time = time.time() - start_time
    memory_increase = profiler.get_memory_increase()
    
    print(f"Model Load Time: {load_time:.2f} seconds")
    print(f"Memory Increase: {memory_increase:.2f} MB")
    
    return {
        'load_time': load_time,
        'memory_increase': memory_increase
    }

@profile
def test_embedding_generation(num_samples=100):
    """Profile memory usage during embedding generation"""
    profiler = MemoryProfiler()
    print(f"\nBaseline Memory Usage: {profiler.start_monitoring():.2f} MB")
    
    # Generate embeddings for sample filenames
    start_time = time.time()
    for i in range(num_samples):
        generate_filename_embedding(f"test_file_{i}.txt")
    
    generation_time = time.time() - start_time
    memory_increase = profiler.get_memory_increase()
    
    print(f"Average Embedding Generation Time: {generation_time/num_samples:.4f} seconds")
    print(f"Memory Increase: {memory_increase:.2f} MB")
    
    return {
        'avg_generation_time': generation_time/num_samples,
        'memory_increase': memory_increase
    }

def generate_report():
    """Generate comprehensive memory usage report"""
    print("=== Memory Usage Report ===")
    print("\nTesting Model Loading...")
    model_stats = test_model_loading()
    
    print("\nTesting Embedding Generation...")
    embedding_stats = test_embedding_generation()
    
    # Save results to file
    with open('memory_profile_results.txt', 'w') as f:
        f.write("=== Memory Profile Results ===\n")
        f.write(f"Model Loading:\n")
        f.write(f"- Load Time: {model_stats['load_time']:.2f} seconds\n")
        f.write(f"- Memory Usage: {model_stats['memory_increase']:.2f} MB\n\n")
        f.write(f"Embedding Generation:\n")
        f.write(f"- Average Generation Time: {embedding_stats['avg_generation_time']:.4f} seconds\n")
        f.write(f"- Memory Usage: {embedding_stats['memory_increase']:.2f} MB\n")

if __name__ == '__main__':
    generate_report()