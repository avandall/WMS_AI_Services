# WMS AI Engine - Modular & Agentic RAG Architecture

A comprehensive AI engine for Warehouse Management Systems featuring advanced RAG capabilities, intelligent agents, and clean modular architecture.

## Features

### Core Capabilities
- **Hybrid Retrieval**: Combines semantic search (vector embeddings) with keyword search (BM25)
- **Advanced RAG Workflows**: Multi-step processing with quality control and fallback mechanisms
- **WMS-Specific Agents**: Database-integrated agents for real-time inventory and order queries
- **Quality Evaluation**: Automated response quality assessment with configurable thresholds
- **Clean Architecture**: Modular design following separation of concerns principles

### Architecture Components
- **Retrieval Module**: Hybrid search with document processing
- **Generation Module**: LLM-based generation with quality evaluation
- **Agents Module**: WMS-specific agents with database tools
- **Workflows Module**: Advanced RAG pipelines with conditional logic
- **Core Engine**: Orchestration layer for all components
- **Configuration**: Centralized settings management
- **Utilities**: Logging, helpers, and validation tools

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env` file (copy from `.env.example`):
```env
# LLM Configuration
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant
EVALUATOR_MODEL=llama-3.3-70b-versatile
TEMPERATURE=0.0

# API Keys
GROQ_API_KEY=your_groq_api_key_here

# Embedding Configuration
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Database Configuration
DB_CONNECTION_STRING=postgresql://wms_user:wms_password@localhost:5433/warehouse_db
VECTOR_DB_PATH=./wms_chroma_db

# Retrieval Configuration
RETRIEVAL_K=3
BM25_WEIGHT=0.4
VECTOR_WEIGHT=0.6

# Evaluation Configuration
QUALITY_THRESHOLD=7.0
```

## Quick Start

### Basic RAG Usage
```python
from ai_engine import WMSEngine, ProcessingMode

# Initialize engine
engine = WMSEngine(mode=ProcessingMode.RAG)

# Add sample data
engine.initialize_sample_data()

# Process queries
result = engine.process_query("What is a Warehouse Management System?")
print(result['response'])
```

### Agent Usage
```python
# Initialize in agent mode for database queries
engine = WMSEngine(mode=ProcessingMode.AGENT)

result = engine.process_query("What is the inventory status for SKU12345?")
print(result['response'])
```

### Hybrid Mode
```python
# Hybrid mode tries RAG first, falls back to agent if needed
engine = WMSEngine(mode=ProcessingMode.HYBRID)
engine.initialize_sample_data()

result = engine.process_query("How does WMS help with order fulfillment?")
print(result['response'])
```

## Architecture Overview

### Directory Structure
```
Services/ai-service/src/ai_engine/
|-- config/           # Configuration management
|   |-- settings.py
|-- models/           # Base models and interfaces
|   |-- base.py
|-- retrieval/        # Document retrieval and processing
|   |-- hybrid_retriever.py
|   |-- document_processor.py
|-- generation/       # Response generation and evaluation
|   |-- llm_generator.py
|   |-- quality_evaluator.py
|-- agents/           # WMS-specific agents
|   |-- wms_agent.py
|-- workflows/        # Advanced RAG workflows
|   |-- rag_workflow.py
|-- core/             # Main engine orchestration
|   |-- engine.py
|-- utils/            # Utilities and helpers
|   |-- logger.py
|   |-- helpers.py
|-- examples/         # Usage examples
|   |-- basic_usage.py
```

### Processing Modes

1. **RAG Mode**: Pure retrieval-augmented generation
2. **Agent Mode**: Database-intelligent agent with tools
3. **Hybrid Mode**: Intelligent switching between RAG and agent

### Advanced Features

#### Quality Control
- Automated response evaluation
- Configurable quality thresholds
- Fallback mechanisms for low-quality responses

#### Hybrid Retrieval
- Semantic search using embeddings
- Keyword search using BM25
- Ensemble retrieval with configurable weights

#### Agent Capabilities
- Real-time database queries
- Inventory status checks
- Order tracking
- Location information

## Examples

See `examples/basic_usage.py` for comprehensive usage examples.

## Configuration

All settings are managed through the `config/settings.py` module and loaded from `.env` file. Key configuration options:

- **LLM Provider**: Groq (only supported provider)
- **Embedding Model**: Configure embedding model for semantic search
- **Database Settings**: PostgreSQL and ChromaDB connections
- **Retrieval Parameters**: K values, weights, thresholds
- **Quality Settings**: Evaluation thresholds and models

**Important**: All configuration values must be set in the `.env` file. No hardcoded defaults are provided.

## Development

The architecture follows clean architecture principles:

1. **Domain Layer**: Core business logic in models
2. **Application Layer**: Use cases and workflows
3. **Infrastructure Layer**: External integrations
4. **Interface Layer**: APIs and user interfaces

## Dependencies

Key dependencies include:
- LangChain ecosystem for LLM integration
- ChromaDB for vector storage
- LangGraph for workflow orchestration
- SQLAlchemy for database operations
- Sentence Transformers for embeddings

## License

This project is part of the WMS (Warehouse Management System) and follows the same licensing terms.
