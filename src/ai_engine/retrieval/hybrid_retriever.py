"""
Hybrid retrieval system combining vector and keyword search
"""
from typing import List
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document as LangchainDocument

from ..models.base import BaseRetriever, Document
from ..config import settings


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining semantic and keyword search"""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': settings.EMBEDDING_DEVICE}
        )
        self.vectorstore = Chroma(
            persist_directory=settings.VECTOR_DB_PATH,
            embedding_function=self.embeddings
        )
        
        # Initialize vector retriever
        self.vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": settings.RETRIEVAL_K}
        )
        
        # Initialize BM25 retriever
        self._setup_bm25_retriever()
        
        # Create ensemble retriever
        if self.bm25_retriever:
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, self.vector_retriever],
                weights=[settings.BM25_WEIGHT, settings.VECTOR_WEIGHT]
            )
        else:
            self.ensemble_retriever = self.vector_retriever
    
    def _setup_bm25_retriever(self):
        """Setup BM25 retriever from existing documents"""
        try:
            all_data = self.vectorstore.get()
            docs_for_bm25 = [
                LangchainDocument(page_content=txt, metadata=meta) 
                for txt, meta in zip(all_data['documents'], all_data['metadatas'])
            ]
            self.bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
            self.bm25_retriever.k = settings.RETRIEVAL_K
        except Exception as e:
            print(f"Warning: Could not setup BM25 retriever: {e}")
            self.bm25_retriever = None
    
    def retrieve(self, query: str, k: int = None) -> List[Document]:
        """Retrieve documents using hybrid search"""
        if k:
            # Update k for both retrievers
            self.vector_retriever.search_kwargs["k"] = k
            if self.bm25_retriever:
                self.bm25_retriever.k = k
        
        docs = self.ensemble_retriever.invoke(query)
        return [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in docs
        ]
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store"""
        langchain_docs = [
            LangchainDocument(page_content=doc.page_content, metadata=doc.metadata)
            for doc in documents
        ]
        
        # Add to vector store
        for i in range(0, len(langchain_docs), 100):
            batch = langchain_docs[i:i + 100]
            self.vectorstore.add_documents(documents=batch)
        
        # Reinitialize BM25 retriever
        self._setup_bm25_retriever()
        
        # Reinitialize ensemble retriever
        if self.bm25_retriever:
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, self.vector_retriever],
                weights=[settings.BM25_WEIGHT, settings.VECTOR_WEIGHT]
            )
