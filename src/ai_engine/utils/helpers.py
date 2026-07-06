"""
Helper utilities for WMS AI Engine
"""
import time
import os
from typing import List, Dict, Any, Optional
from pathlib import Path


def ensure_directory_exists(path: str) -> None:
    """Ensure directory exists, create if it doesn't"""
    Path(path).mkdir(parents=True, exist_ok=True)


def format_documents_for_display(documents: List[str], max_length: int = 500) -> str:
    """Format documents for display with truncation"""
    formatted_docs = []
    for i, doc in enumerate(documents, 1):
        truncated = doc[:max_length] + "..." if len(doc) > max_length else doc
        formatted_docs.append(f"Document {i}:\n{truncated}")
    return "\n\n".join(formatted_docs)


def retry_with_backoff(func, max_retries: int = 3, backoff_factor: float = 1.0):
    """Retry function with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            wait_time = backoff_factor * (2 ** attempt)
            time.sleep(wait_time)


def validate_api_keys() -> Dict[str, bool]:
    """Validate required API keys are present"""
    from ..config import settings
    
    validation_results = {
        "groq": bool(settings.GROQ_API_KEY)
    }
    
    return validation_results


def sanitize_text(text: str) -> str:
    """Sanitize text for processing"""
    # Remove excessive whitespace
    text = ' '.join(text.split())
    # Remove potentially harmful characters (basic sanitization)
    dangerous_chars = ['<', '>', '&', '"', "'"]
    for char in dangerous_chars:
        text = text.replace(char, '')
    return text


def calculate_retrieval_metrics(retrieved_docs: List[str], relevant_docs: List[str]) -> Dict[str, float]:
    """Calculate basic retrieval metrics"""
    if not relevant_docs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # Simple exact match for demonstration
    retrieved_set = set(retrieved_docs)
    relevant_set = set(relevant_docs)
    
    true_positives = len(retrieved_set & relevant_set)
    false_positives = len(retrieved_set - relevant_set)
    false_negatives = len(relevant_set - retrieved_set)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def create_wms_sample_data() -> List[Dict[str, Any]]:
    """Create realistic WMS data based on actual database schema"""
    return [
        {
            "text": "Warehouse Management System (WMS) manages multiple warehouses with storage positions organized by shelf codes (A-01, A-02, B-01, B-02). Each position has type STORAGE and is linked to specific warehouse locations. The system tracks inventory at three levels: total inventory, warehouse-level inventory, and position-level inventory.",
            "metadata": {"source": "wms_docs", "category": "warehouse_structure"}
        },
        {
            "text": "Product management in WMS includes product_id (2001-2010), product name, description, and pricing. Each product has inventory tracking with quantity monitoring across different warehouses and positions. Products can be moved between locations through import/export documents.",
            "metadata": {"source": "wms_docs", "category": "product_management"}
        },
        {
            "text": "Document system handles IMPORT and EXPORT transactions with POSTED status. Each document has source/destination warehouse, customer information, and contains multiple document items with quantities and unit prices. Documents are created by users and include timestamps for tracking.",
            "metadata": {"source": "wms_docs", "category": "document_management"}
        },
        {
            "text": "Customer management includes company name, email, phone, address, and debt balance tracking. Each customer can be linked to multiple documents for import/export transactions. The system maintains customer relationships and financial records.",
            "metadata": {"source": "wms_docs", "category": "customer_management"}
        },
        {
            "text": "User roles in WMS include admin, warehouse manager, and regular users. Admin users have full system access, warehouse managers handle operations, and regular users have limited permissions. All users have email authentication and active status tracking.",
            "metadata": {"source": "wms_docs", "category": "user_management"}
        },
        {
            "text": "Inventory flow: Products start with total quantity (100 units), distributed to warehouses (100 units per warehouse), then further distributed to specific positions (100 units per position). This three-tier inventory system ensures accurate tracking at each level.",
            "metadata": {"source": "wms_docs", "category": "inventory_flow"}
        },
        {
            "text": "Position inventory tracking links specific positions to products with quantity monitoring. Each position belongs to a warehouse and stores specific product quantities. This enables precise location-based inventory management and picking operations.",
            "metadata": {"source": "wms_docs", "category": "position_inventory"}
        },
        {
            "text": "Audit logging system tracks all changes with audit_logs and audit_events tables. This maintains comprehensive records of system operations, user actions, and data modifications for compliance and troubleshooting purposes.",
            "metadata": {"source": "wms_docs", "category": "audit_system"}
        },
        {
            "text": "Document items represent individual product transactions within documents. Each item includes product_id, quantity, and unit_price, allowing for detailed tracking of specific product movements and financial calculations.",
            "metadata": {"source": "wms_docs", "category": "document_items"}
        },
        {
            "text": "Warehouse inventory provides intermediate tracking between total inventory and position inventory. Each warehouse maintains its own product quantities, enabling regional inventory management and transfer operations between warehouses.",
            "metadata": {"source": "wms_docs", "category": "warehouse_inventory"}
        }
    ]
