import unittest
import os
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from pathlib import Path
from datetime import datetime
from django.test import TestCase
from django.conf import settings
from transformers import AutoModel, AutoTokenizer
from .utils import (
    setup_model_directory,
    init_embedding_model,
    init_arabic_model,
    generate_arabic_embedding,
    mean_pooling,
    get_file_metadata,
    generate_filename_embedding,
    search_similar_files,
    scan_directory,
    get_common_directories,
    update_indexed_files,
    get_search_agent,
    search_files_with_rag
)
from .models import Indexer
from .search_agent import SearchAgentService

class UtilsTestCase(TestCase):
    def setUp(self):
        """Set up test environment"""
        # temp file 
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

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModel.from_pretrained')
    def test_init_arabic_model(self, mock_model, mock_tokenizer):
        """Test Arabic model initialization"""
        # Mock the model and tokenizer
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        
        init_arabic_model()
        
        # Check if model and tokenizer were initialized with correct parameters
        mock_model.assert_called_once_with(
            settings.DEFAULT_EMBEDDING_ARABIC,
            cache_dir=settings.DEFAULT_EMBEDDING_MODEL_PATH_ARABIC
        )
        mock_tokenizer.assert_called_once_with(
            settings.DEFAULT_EMBEDDING_ARABIC,
            cache_dir=settings.DEFAULT_EMBEDDING_MODEL_PATH_ARABIC
        )

    @patch('indexer.utils._arabic_model')
    @patch('indexer.utils._arabic_tokenizer')
    def test_generate_arabic_embedding(self, mock_tokenizer, mock_model):
        """Test Arabic embedding generation"""
        # Mock tokenizer output
        mock_tokenizer.return_value = {
            'input_ids': torch.ones((1, 10)),
            'attention_mask': torch.ones((1, 10))
        }
        
        # Mock model output
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.ones((1, 10, 768))  # Common BERT hidden size
        mock_model.return_value = mock_output
        
        # Test embedding generation
        test_text = "اختبار النص العربي"
        embedding = generate_arabic_embedding(test_text)
        
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.dtype, np.float32)
        self.assertEqual(len(embedding.shape), 1)  # Should be a 1D array

    def test_mean_pooling(self):
        """Test mean pooling function"""
        # Create sample inputs
        token_embeddings = torch.ones((2, 3, 4))  # batch_size=2, seq_len=3, hidden_size=4
        attention_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])  # Mask out some tokens
        
        # Calculate mean pooling
        pooled = mean_pooling(token_embeddings, attention_mask)
        
        self.assertEqual(pooled.shape, (2, 4))  # Should be (batch_size, hidden_size)
        # Check if masked tokens are properly handled
        self.assertTrue(torch.allclose(pooled[0], torch.ones(4) * 1.0))  # First sequence has 2 tokens
        self.assertTrue(torch.allclose(pooled[1], torch.ones(4) * 1.0))  # Second sequence has 1 token

    def test_generate_filename_embedding_with_arabic(self):
        """Test filename embedding generation with Arabic model option"""
        # Test with Arabic model
        arabic_filename = "ملف_تجريبي.txt"
        arabic_embedding = generate_filename_embedding(arabic_filename, use_arabic_model=True)
        self.assertIsInstance(arabic_embedding, np.ndarray)
        self.assertEqual(arabic_embedding.dtype, np.float32)
        
        # Test with default model
        english_filename = "test_file.txt"
        english_embedding = generate_filename_embedding(english_filename, use_arabic_model=False)
        self.assertIsInstance(english_embedding, np.ndarray)
        self.assertEqual(english_embedding.dtype, np.float32)

class SearchAgentTestCase(TestCase):
    """Test cases for RAG-powered search functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_files = [
            Indexer.objects.create(
                file_name="document.pdf",
                file_path="/test/path/document.pdf",
                file_type="pdf",
                creation_date=datetime.now(),
                size=1024
            ),
            Indexer.objects.create(
                file_name="image.jpg",
                file_path="/test/path/image.jpg",
                file_type="jpg",
                creation_date=datetime.now(),
                size=2048
            )
        ]
        
        # Generate and set embeddings for test files
        for file in self.test_files:
            embedding = np.random.rand(1024).astype(np.float32)
            file.set_embedding(embedding)
            file.save()

    def tearDown(self):
        """Clean up after tests"""
        Indexer.objects.all().delete()

    @patch('indexer.search_agent.CodeAgent')
    def test_search_agent_initialization(self, mock_code_agent):
        """Test SearchAgentService initialization"""
        agent = SearchAgentService()
        self.assertIsNotNone(agent)
        mock_code_agent.assert_called_once()

    def test_get_search_agent_singleton(self):
        """Test search agent singleton pattern"""
        with patch('indexer.search_agent.SearchAgentService') as mock_service:
            agent1 = get_search_agent()
            agent2 = get_search_agent()
            
            # Should only be initialized once
            mock_service.assert_called_once()
            self.assertEqual(agent1, agent2)

    @patch('indexer.search_agent.SearchAgentService')
    def test_search_files_with_rag(self, mock_service):
        """Test RAG-powered file search"""
        # Mock the search results
        mock_results = [
            {
                'id': 1,
                'file_name': 'document.pdf',
                'file_path': '/test/path/document.pdf',
                'file_type': 'pdf',
                'creation_date': datetime.now(),
                'size': '1.00 KB'
            }
        ]
        
        mock_instance = mock_service.return_value
        mock_instance.search_files.return_value = mock_results
        
        # Test search
        results = search_files_with_rag("find pdf documents", top_n=1)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['file_name'], 'document.pdf')
        mock_instance.search_files.assert_called_once_with("find pdf documents", 1)

    @patch('indexer.search_agent.SearchAgentService')
    def test_search_files_with_rag_fallback(self, mock_service):
        """Test fallback search when RAG model is unavailable"""
        # Mock SearchAgentService to raise an exception
        mock_service.side_effect = Exception("Model failed to load")
        
        # Test search with fallback
        results = search_files_with_rag("document", top_n=5)
        
        # Should still return results using basic search
        self.assertIsInstance(results, list)
        for result in results:
            self.assertIn('note', result)
            self.assertEqual(result['note'], 'Basic search only (RAG model unavailable)')

    def test_query_database_tool(self):
        """Test the database query tool used by smolagents"""
        from indexer.search_agent import query_database
        
        # Test valid SQL query
        result = query_database("SELECT COUNT(*) FROM indexer_indexer")
        self.assertIsInstance(result, str)
        
        # Test invalid SQL query
        result = query_database("INVALID SQL")
        self.assertIn("Error executing query", result)

if __name__ == '__main__':
    unittest.main()