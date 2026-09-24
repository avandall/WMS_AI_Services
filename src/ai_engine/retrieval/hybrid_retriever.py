"""
Hybrid retrieval system combining distributed Qdrant vector store and keyword search (BM25)
"""
import os
import json
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document as LangchainDocument

from ..models.base import BaseRetriever, Document
from ..config import settings


class HybridRetriever(BaseRetriever):
    """Distributed hybrid retriever combining Qdrant vector search and BM25 text search"""
    
    def __init__(self, bm25_store_path: Optional[str] = None, qdrant_client: Optional[QdrantClient] = None):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': settings.EMBEDDING_DEVICE}
        )
        
        # Initialize distributed Qdrant vector store (Twelve-Factor Attached Resource)
        self.collection_name = settings.QDRANT_COLLECTION
        if qdrant_client:
            self.qdrant_client = qdrant_client
        else:
            try:
                # Try connecting to remote/distributed Qdrant service
                self.qdrant_client = QdrantClient(
                    url=settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY or None,
                    timeout=2.0,
                    check_compatibility=False,
                )
                # Test connection
                self.qdrant_client.get_collections()
            except Exception:
                # Fallback to in-memory Qdrant instance for test isolation / stateless run
                self.qdrant_client = QdrantClient(":memory:")
        
        # Ensure collection exists
        try:
            collections = [c.name for c in self.qdrant_client.get_collections().collections]
            if self.collection_name not in collections:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )
        except Exception:
            pass

        self.vectorstore = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
        
        # Safe JSON persistence for BM25 docs to prevent RCE (No pickle)
        default_bm25 = os.path.join(settings.VECTOR_DB_PATH, "bm25_store.json")
        self.bm25_store_path = bm25_store_path or default_bm25
        self._bm25_docs: List[LangchainDocument] = []
        
        # Initialize vector retriever
        self.vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": settings.RETRIEVAL_K}
        )
        
        # Initialize BM25 retriever from persistent store
        self._setup_bm25_retriever()
        
        # Create ensemble retriever
        self._update_ensemble_retriever()
    
    def _setup_bm25_retriever(self):
        """Setup BM25 retriever from persistent document store on disk safely with JSON"""
        try:
            if os.path.exists(self.bm25_store_path):
                with open(self.bm25_store_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                
                self._bm25_docs = [
                    LangchainDocument(
                        page_content=item.get("page_content", ""),
                        metadata=item.get("metadata", {})
                    )
                    for item in raw_data
                ]
                
                if self._bm25_docs:
                    self.bm25_retriever = BM25Retriever.from_documents(self._bm25_docs)
                    self.bm25_retriever.k = settings.RETRIEVAL_K
                else:
                    self.bm25_retriever = None
            else:
                self._bm25_docs = []
                self.bm25_retriever = None
        except Exception as e:
            print(f"Warning: Could not load persistent BM25 index: {e}")
            self._bm25_docs = []
            self.bm25_retriever = None
    
    def _update_ensemble_retriever(self):
        """Update ensemble retriever combining BM25 and vector retriever"""
        if self.bm25_retriever:
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, self.vector_retriever],
                weights=[settings.BM25_WEIGHT, settings.VECTOR_WEIGHT]
            )
        else:
            self.ensemble_retriever = self.vector_retriever

    def _persist_bm25_docs(self):
        """Save BM25 documents corpus to persistent store on disk using secure JSON (zero pickle)"""
        try:
            store_dir = os.path.dirname(self.bm25_store_path)
            if store_dir:
                os.makedirs(store_dir, exist_ok=True)
            serializable = [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in self._bm25_docs
            ]
            with open(self.bm25_store_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Could not persist BM25 docs to disk: {e}")

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Retrieve documents using hybrid search"""
        if k:
            self.vector_retriever.search_kwargs["k"] = k
            if self.bm25_retriever:
                self.bm25_retriever.k = k
        
        docs = self.ensemble_retriever.invoke(query)
        return [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in docs
        ]
    
    def add_documents(self, documents: List[Document]):
        """Add documents to vector store and update persistent BM25 index"""
        langchain_docs = [
            LangchainDocument(page_content=doc.page_content, metadata=doc.metadata)
            for doc in documents
        ]
        
        # Add to vector store in batches
        for i in range(0, len(langchain_docs), 100):
            batch = langchain_docs[i:i + 100]
            self.vectorstore.add_documents(documents=batch)
        
        # Incrementally update BM25 document corpus & persistent store
        self._bm25_docs.extend(langchain_docs)
        if self._bm25_docs:
            self.bm25_retriever = BM25Retriever.from_documents(self._bm25_docs)
            self.bm25_retriever.k = settings.RETRIEVAL_K
        
        self._persist_bm25_docs()
        
        # Reinitialize ensemble retriever
        self._update_ensemble_retriever()
