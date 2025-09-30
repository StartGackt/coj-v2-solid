# Thai Legal NLP Analysis Service

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive FastAPI-based application for Natural Language Processing (NLP) tasks on Thai legal documents. Built with SOLID principles for maintainability and scalability, featuring advanced AI models for legal text analysis, question answering, and semantic search.

## ✨ Features

- **🔍 Legal Document Analysis**: Extract obligations, exceptions, and compliance steps from Thai legal articles
- **🤖 NLP Pipeline**: Named Entity Recognition (NER), text classification, and semantic search
- **📊 Knowledge Graph**: Neo4j-powered graph database for legal entity relationships
- **❓ Question Answering**: Fine-tuned models for legal Q&A based on context
- **🔎 Semantic Search**: Meaning-based document retrieval using sentence transformers
- **🗄️ Database Integration**: PostgreSQL for structured data + Neo4j for knowledge graphs
- **🔄 Model Fine-tuning**: Scripts to improve AI models on legal domain data
- **📈 Monitoring**: Health checks and logging for production deployment

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Services      │    │   Repositories  │
│   Endpoints     │◄──►│   Business      │◄──►│   Data Access   │
│                 │    │   Logic         │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │     Neo4j       │    │   AI Models     │
│   Structured    │    │   Knowledge     │    │   (Transformers)│
│   Data          │    │   Graph         │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- **Python**: 3.11 or higher
- **Databases**:
  - PostgreSQL 13+ (for structured data)
  - Neo4j 5.0+ (for knowledge graphs)
- **Package Manager**: uv (recommended) or pip
- **Memory**: 8GB+ RAM recommended for model training

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/StartGackt/coj-v2-solid.git
cd coj-v2-solid
```

### 2. Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e .
```

### 3. Database Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your database credentials
nano .env
```

**Required Environment Variables:**

```env
# PostgreSQL
DATABASE_URL=postgresql://user:password@localhost:5432/legal_nlp

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Optional: OpenAI API for enhanced features
OPENAI_API_KEY=your_key_here
```

### 4. Database Initialization

```bash
# Run PostgreSQL migrations
alembic upgrade head

# Ensure Neo4j is running and accessible
```

### 5. Start the Application

```bash
python main.py
```

Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

## 📖 Usage

### API Endpoints

#### Legal Article Analysis

```bash
curl -X POST "http://localhost:8000/api/v1/legal-ontology/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "article_number": "มาตรา 12",
    "summary": "ห้ามนายจ้างเรียกหลักประกันจากลูกจ้าง ยกเว้นตำแหน่งที่ต้องรับผิดชอบด้านการเงิน",
    "obligations": [
      {
        "actor": "นายจ้าง",
        "action": "ต้องไม่เรียกหรือรับหลักประกันจากลูกจ้าง",
        "timeline": null
      }
    ],
    "exceptions": [
      {
        "description": "อนุญาตให้เรียกหลักประกันเมื่อหน้าที่ลูกจ้างเกี่ยวข้องกับการเงิน"
      }
    ]
  }'
```

#### Question Answering

```bash
curl -X POST "http://localhost:8000/api/v1/qa/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "ห้ามนายจ้างเรียกหลักประกันจากใคร",
    "context": "มาตรา 12 ห้ามมิให้นายจ้างเรียกหรือรับหลักประกันเพื่อการใดๆ จากลูกจ้าง"
  }'
```

#### Semantic Search

```bash
curl -X POST "http://localhost:8000/api/v1/search/semantic" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "กฎหมายแรงงาน",
    "limit": 10
  }'
```

### Training Scripts

#### Train Text Classifier

```bash
python scripts/train_text_classifier.py
```

#### Fine-tune QA Model

```bash
python scripts/finetune_qa_model.py
```

#### Fine-tune Semantic Search

```bash
python scripts/finetune_semantic_search.py
```

This script now consumes curated triplets from `app/dataset/semantic_triplets.json`. Each entry defines an anchor query, the most relevant legal article (`positive`), and optional `hard_negatives` that the model should learn to distinguish. You can expand the dataset by appending new triplets that reference articles available in `app/dataset/data1.txt`. During fine-tuning the script augments each triplet with extra random negatives to produce a rich training set.

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=app --cov-report=html

# Run specific test file
uv run pytest tests/test_legal_article_analysis.py -v
```

## 🔧 Development

### Project Structure

```
coj-v2-solid/
├── alembic/              # Database migrations
├── app/
│   ├── api/             # FastAPI route handlers
│   ├── core/            # Configuration & settings
│   ├── db/              # Database models & connections
│   ├── models/          # Pydantic schemas
│   ├── nlp/             # NLP processing modules
│   ├── repositories/    # Data access layer
│   └── services/        # Business logic
├── models/              # Trained AI models
├── scripts/             # Training & utility scripts
├── tests/               # Test suite
└── main.py              # Application entry point
```

### Adding New Features

1. **API Endpoints**: Add routes in `app/api/v1/endpoints/`
2. **Business Logic**: Implement services in `app/services/`
3. **Data Models**: Define schemas in `app/models/`
4. **Database Changes**: Create migrations with `alembic revision`

### Code Quality

```bash
# Format code
uv run black .

# Lint code
uv run flake8 .

# Type checking
uv run mypy .
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow SOLID principles
- Write tests for new features
- Update documentation
- Use type hints
- Keep commits atomic and descriptive

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/) and [Neo4j](https://neo4j.com/)
- NLP powered by [Hugging Face Transformers](https://huggingface.co/)
- Inspired by Thai legal document processing challenges

## 📞 Support

For questions or issues:

- Open an [issue](https://github.com/StartGackt/coj-v2-solid/issues) on GitHub
- Check the [documentation](https://github.com/StartGackt/coj-v2-solid/wiki)

---

**Note**: This project is designed for Thai legal text processing. For other languages, model fine-tuning may be required.
