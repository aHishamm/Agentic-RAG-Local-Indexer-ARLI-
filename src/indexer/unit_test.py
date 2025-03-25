import unittest
import os
import numpy as np
from pathlib import Path
from datetime import datetime
from django.test import TestCase
from django.conf import settings
from .utils import (
    setup_model_directory,
    init_embedding_model,
    get_file_metadata,
    generate_filename_embedding,
    search_similar_files,
    scan_directory,
    get_common_directories,
    update_indexed_files
)
from .models import Indexer

class UtilsTestCase(TestCase):
    def setUp(self):
        """Set up test environment"""
        # temp fiel 
        self.test_file_path = "test_file.txt"
        with open(self.test_file_path, "w") as f:
            f.write("Test content")
        
        # Create test instances 
        self.test_indexer = Indexer.objects.create(
            file_name="test_file.txt",
            file_path="/test/path/test_file.txt",
            file_type="txt",
            creation_date=datetime.now(),
            size=100
        )
        
        # Generate and set embedding for the test file
        embedding = np.random.rand(1024).astype(np.float32)  # Default embedding size for UAE-Large-V1
        self.test_indexer.set_embedding(embedding)
        self.test_indexer.save()

    def tearDown(self):
        """Clean up after tests"""
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)
        Indexer.objects.all().delete()

    def test_setup_model_directory(self):
        """Test if model directory is created correctly"""
        models_dir = setup_model_directory()
        self.assertTrue(os.path.exists(models_dir))
        self.assertTrue(os.path.isdir(models_dir))
        self.assertEqual(os.path.basename(models_dir), 'HF_Models')

    def test_init_embedding_model(self):
        """Test model initialization"""
        # Test default model
        default_model = init_embedding_model()
        self.assertIsNotNone(default_model)
        
        # Test Arabic model
        arabic_model = init_embedding_model(settings.DEFAULT_EMBEDDING_ARABIC)
        self.assertIsNotNone(arabic_model)

    def test_get_file_metadata(self):
        """Test file metadata extraction"""
        metadata = get_file_metadata(self.test_file_path)
        
        self.assertEqual(metadata['file_name'], 'test_file.txt')
        self.assertTrue(os.path.exists(metadata['file_path']))
        self.assertEqual(metadata['file_type'], 'txt')
        self.assertIsInstance(metadata['creation_date'], datetime)
        self.assertGreater(metadata['size'], 0)

    def test_generate_filename_embedding(self):
        """Test filename embedding generation"""
        # Test default model embedding
        embedding = generate_filename_embedding("test_file.txt")
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.dtype, np.float32)
        
        # Test Arabic model embedding
        arabic_embedding = generate_filename_embedding("test_file.txt", use_arabic_model=True)
        self.assertIsInstance(arabic_embedding, np.ndarray)
        self.assertEqual(arabic_embedding.dtype, np.float32)

    def test_search_similar_files(self):
        """Test similar files search functionality"""
        results = search_similar_files("test", [self.test_indexer])
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertIn('file_name', results[0])
        self.assertIn('file_path', results[0])
        self.assertIn('similarity', results[0])
        self.assertIsInstance(results[0]['similarity'], float)

    def test_scan_directory(self):
        """Test directory scanning"""
        # Create a temporary test directory with files
        test_dir = "test_dir"
        os.makedirs(test_dir, exist_ok=True)
        test_files = ["file1.txt", "file2.txt"]
        
        for file_name in test_files:
            with open(os.path.join(test_dir, file_name), "w") as f:
                f.write("Test content")
        
        try:
            results = scan_directory(test_dir)
            self.assertEqual(len(results), len(test_files))
            for result in results:
                self.assertIn('file_name', result)
                self.assertIn('embedding', result)
        finally:
            # Clean up
            for file_name in test_files:
                try:
                    os.remove(os.path.join(test_dir, file_name))
                except:
                    pass
            try:
                os.rmdir(test_dir)
            except:
                pass

    def test_get_common_directories(self):
        """Test common directories retrieval"""
        dirs = get_common_directories()
        self.assertIsInstance(dirs, list)
        for dir_path in dirs:
            self.assertTrue(os.path.exists(dir_path))

    def test_update_indexed_files(self):
        """Test database update with indexed files"""
        # Create a test directory with a file
        test_dir = "test_dir"
        os.makedirs(test_dir, exist_ok=True)
        test_file = os.path.join(test_dir, "test_file.txt")
        
        with open(test_file, "w") as f:
            f.write("Test content")
        
        try:
            # Patch get_common_directories to return our test directory
            def mock_get_common_directories():
                return [test_dir]
            
            original_func = get_common_directories
            try:
                globals()['get_common_directories'] = mock_get_common_directories
                update_indexed_files()
                
                # Check if file was indexed
                self.assertTrue(
                    Indexer.objects.filter(file_name="test_file.txt").exists()
                )
            finally:
                globals()['get_common_directories'] = original_func
        finally:
            # Clean up
            try:
                os.remove(test_file)
                os.rmdir(test_dir)
            except:
                pass

if __name__ == '__main__':
    unittest.main()