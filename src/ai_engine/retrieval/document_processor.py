"""
Document processing and chunking for WMS RAG system
"""
import sys
import os
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document as LangchainDocument
import bs4
from ai_engine.models.base import Document


class DocumentProcessor:
    """Handles document loading and processing"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
    
    def load_from_web(self, urls: List[str]) -> List[Document]:
        """Load documents from web URLs"""
        loader = WebBaseLoader(
            web_paths=urls,
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        docs = loader.load()
        return self._process_documents(docs)
    
    def load_from_text(self, texts: List[str], metadatas: List[dict] = None) -> List[Document]:
        """Load documents from raw text"""
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        docs = [
            LangchainDocument(page_content=text, metadata=metadata)
            for text, metadata in zip(texts, metadatas)
        ]
        return self._process_documents(docs)
    
    def _process_documents(self, docs: List[LangchainDocument]) -> List[Document]:
        """Process and chunk documents"""
        chunks = self.text_splitter.split_documents(docs)
        return [
            Document(page_content=chunk.page_content, metadata=chunk.metadata)
            for chunk in chunks
        ]
