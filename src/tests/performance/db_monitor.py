import os
import time
import sys
import psutil
from django.db import connection
from django.db.backends.utils import CursorWrapper
from functools import wraps
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from indexer.models import Indexer
from django.conf import settings

class DBPerformanceMonitor:
    def __init__(self):
        self.query_times = []
        self.connection_count = 0
        self.start_time = None

    def start_monitoring(self):
        """Start monitoring database performance"""
        self.start_time = time.time()
        self.connection_count = len(connection.connection.connection.get_autocommit())
        return self

    def record_query(self, duration):
        """Record a query execution time"""
        self.query_times.append(duration)

    def get_stats(self):
        """Get database performance statistics"""
        if not self.query_times:
            return {
                'avg_query_time': 0,
                'max_query_time': 0,
                'total_queries': 0,
                'active_connections': self.connection_count
            }

        return {
            'avg_query_time': sum(self.query_times) / len(self.query_times),
            'max_query_time': max(self.query_times),
            'total_queries': len(self.query_times),
            'active_connections': self.connection_count
        }

def measure_query_time(func):
    """Decorator to measure query execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        print(f"Query '{func.__name__}' took {duration:.4f} seconds")
        return result
    return wrapper

class QueryProfiler:
    @measure_query_time
    def test_basic_queries(self):
        """Test basic database operations"""
        # Test SELECT
        list(Indexer.objects.all()[:10])
        
        # Test COUNT
        Indexer.objects.count()
        
        # Test filtering
        list(Indexer.objects.filter(file_type='txt')[:10])

    @measure_query_time
    def test_complex_queries(self):
        """Test more complex database operations"""
        # Test aggregation
        from django.db.models import Count
        Indexer.objects.values('file_type').annotate(count=Count('id'))
        
        # Test joins if we add related models in future

def generate_db_report():
    """Generate database performance report"""
    monitor = DBPerformanceMonitor()
    monitor.start_monitoring()
    
    profiler = QueryProfiler()
    print("\n=== Database Performance Test ===")
    
    # Run test queries
    print("\nTesting Basic Queries...")
    profiler.test_basic_queries()
    
    print("\nTesting Complex Queries...")
    profiler.test_complex_queries()
    
    # Get final stats
    stats = monitor.get_stats()
    
    # Save results
    with open('db_performance_results.txt', 'w') as f:
        f.write("=== Database Performance Results ===\n")
        f.write(f"Average Query Time: {stats['avg_query_time']:.4f} seconds\n")
        f.write(f"Maximum Query Time: {stats['max_query_time']:.4f} seconds\n")
        f.write(f"Total Queries Run: {stats['total_queries']}\n")
        f.write(f"Active Connections: {stats['active_connections']}\n")

if __name__ == '__main__':
    generate_db_report()