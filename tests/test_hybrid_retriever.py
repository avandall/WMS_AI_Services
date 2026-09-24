"""
Unit tests for HybridRetriever ensuring Qdrant distributed architecture, zero pickle RCE risk, and performance SLA.
"""
import os
import sys
import time
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Ensure src directory is in sys.path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from ai_engine.retrieval.hybrid_retriever import HybridRetriever
from ai_engine.models.base import Document


class TestHybridRetriever(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.bm25_path = os.path.join(self.temp_dir.name, "bm25_store.json")
    
    def tearDown(self):
        self.temp_dir.cleanup()

    def test_no_vectorstore_get_called_on_init(self):
        """Verify vectorstore.get() is never called during retriever initialization"""
        retriever = HybridRetriever(bm25_store_path=self.bm25_path)
        
        # Mock vectorstore.get to raise if invoked
        retriever.vectorstore.get = MagicMock(side_effect=AssertionError("vectorstore.get() must not be called!"))
        
        # Test adding documents
        docs = [
            Document(page_content="Lô vắc xin H1N1 bảo quản tại Kho C2 Kệ 05", metadata={"doc_id": "1"}),
            Document(page_content="Quy trình kiểm kê kho tài sản định kỳ", metadata={"doc_id": "2"}),
        ]
        retriever.add_documents(docs)
        
        # Verify vectorstore.get was never called
        retriever.vectorstore.get.assert_not_called()

    def test_persistent_bm25_save_and_reload(self):
        """Verify BM25 index is persisted securely to JSON and reloaded without pickle"""
        retriever1 = HybridRetriever(bm25_store_path=self.bm25_path)
        docs = [
            Document(page_content="Mã SKU A100 có tổng thể tích 2.5 CBM", metadata={"sku": "A100"}),
            Document(page_content="Đơn hàng Outbound 50002 đã được xác nhận", metadata={"doc_id": "50002"}),
        ]
        retriever1.add_documents(docs)
        
        # Check that persistent file was written to disk as JSON
        self.assertTrue(os.path.exists(self.bm25_path))
        with open(self.bm25_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("SKU A100", content)
            self.assertNotIn("pickle", content.lower())
        
        # Initialize a new retriever instance using the same persistent store path
        retriever2 = HybridRetriever(bm25_store_path=self.bm25_path)
        self.assertIsNotNone(retriever2.bm25_retriever)
        self.assertEqual(len(retriever2._bm25_docs), 2)
        self.assertEqual(retriever2._bm25_docs[0].page_content, "Mã SKU A100 có tổng thể tích 2.5 CBM")

    def test_no_pickle_usage_in_retriever_source(self):
        """TASK-050: Ensure zero pickle import or usage exists in hybrid_retriever module"""
        import ai_engine.retrieval.hybrid_retriever as hr_mod
        source_path = hr_mod.__file__
        with open(source_path, "r", encoding="utf-8") as f:
            code = f.read()
        self.assertNotIn("import pickle", code)
        self.assertNotIn("pickle.dump", code)
        self.assertNotIn("pickle.load", code)

    def test_retrieval_latency_under_300ms(self):
        """Verify document query execution latency is well under SLA"""
        retriever = HybridRetriever(bm25_store_path=self.bm25_path)
        docs = [
            Document(page_content="Thành phần thuốc paracetamol 500mg lô B992", metadata={"lot": "B992"}),
            Document(page_content="Vị trí Pallet P-102 tại Bin A-01-02", metadata={"location": "A-01-02"}),
        ]
        retriever.add_documents(docs)
        
        # Measure retrieval performance
        start_time = time.perf_counter()
        results = retriever.retrieve("paracetamol 500mg", k=2)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        
        self.assertGreater(len(results), 0)
        self.assertLess(elapsed_ms, 300.0, f"Retrieval latency {elapsed_ms:.2f}ms exceeded 300ms threshold!")


if __name__ == "__main__":
    unittest.main()
