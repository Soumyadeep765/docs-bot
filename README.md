#  Telegram Bot API Documentation Search

A **high-performance REST API** for searching **Telegram Bot API documentation** with lightning-fast response times.  
This service parses official Telegram documentation and provides an efficient search interface with advanced query capabilities.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Production-brightgreen)

---

## Features

- Ultra-fast search with in-memory caching and optimized algorithms  
- Multiple data sources: Bot API, WebApps, Features, FAQ, Tutorials  
- Advanced search syntax with special operators  
- Unique ID system for entities and fields  
- Multiple output formats: JSON, HTML, Markdown  
- Manual documentation refresh  
- Memory + database caching with configurable TTL  

---

##  API Endpoints

### Core Search

- `GET /api/search` – Search documentation  
- `GET /api/lookup/id/{entity_id}` – Lookup entity by ID  
- `GET /api/lookup/field/{field_id}` – Lookup field by ID  
- `GET /api/entity/{entity_name}` – Get entity by name  

### Management

- `POST /api/update` – Force documentation refresh  
- `GET /api/stats` – System statistics  
- `POST /api/cache/clear` – Clear memory cache  
- `GET /health` – Health check  

---

##  Advanced Search Syntax

### Type Filtering

```bash
!method sendMessage
!object User
```

### Property Search

```bash
.message_id
.chat_id
```

### Wildcard Search

```bash
send*
get_*
chat_*
```

### Combined Queries

```bash
!method .*photo
!object .from
```

---

## 🛠 Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
git clone https://github.com/Soumyadeep765/docs-bot.git
cd docs-bot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

### Docker

```bash
docker build -t docs-bot .
docker run -p 5000:5000 docs-bot
```

---

## API Usage Examples

```bash
curl "http://localhost:5000/api/search?q=sendMessage"
```

```bash
curl "http://localhost:5000/api/search?q=!method%20sendMessage&advanced=true"
```

---

##  Response Example

```json
{
  "query": "sendMessage",
  "count": 1,
  "results": [
    {
      "id": "a1b2c3d4",
      "name": "sendMessage",
      "type": "method",
      "description": "Sends text messages...",
      "fields": [
        {
          "id": "a1b2c3d4_1",
          "name": "chat_id",
          "type": "Integer or String",
          "required": true
        }
      ],
      "reference": "https://core.telegram.org/bots/api#sendmessage",
      "score": 1000
    }
  ],
  "search_time_ms": 0.45
}
```

---

## Configuration

```bash
DB_FILE=/path/to/database.db
CACHE_DURATION=1800
PORT=5000
LOG_LEVEL=INFO
```

---

##  Performance

- Search: < 5ms  
- Entity lookup: < 2ms  
- Startup load: ~2–3 seconds  

---

##  Architecture

**Data Flow**

1. Fetch official Telegram docs  
2. Parse HTML into entities  
3. Assign unique IDs  
4. Store in SQLite (FTS enabled)  
5. Cache entire dataset in memory  
6. Serve queries from memory  

---

##  Contributing

1. Fork the repository  
2. Create a feature branch  
3. Commit changes  
4. Add tests if needed  
5. Submit a pull request  

---

##  License

MIT License. See `LICENSE` for details.

---

##  Acknowledgments

- Telegram Documentation Team  
- FastAPI Team  
- Open-source contributors  

---

Built with performance in mind ⚡  
Search the Telegram Bot API documentation faster than ever.
