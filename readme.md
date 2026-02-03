# Telegram Bot API Documentation Search

A high-performance REST API for searching Telegram Bot API documentation with lightning-fast response times. This service parses official Telegram documentation and provides an efficient search interface with advanced query capabilities.

https://img.shields.io/badge/Python-3.10%2B-blue
https://img.shields.io/badge/FastAPI-0.104%2B-green
https://img.shields.io/badge/License-MIT-yellow
https://img.shields.io/badge/Status-Production-brightgreen

## Features

· Ultra-fast search: In-memory caching and optimized algorithms for sub-millisecond response times
· Multiple data sources: Integrates documentation from Bot API, WebApps, Features, FAQ, and Tutorial
· Advanced search syntax: Special operators for precise filtering
· Unique ID system: Every entity and field has a permanent, searchable identifier
· Multiple output formats: JSON, HTML, and Markdown responses
· Real-time updates: Manual trigger to refresh documentation data
· Comprehensive caching: Memory and database caching with configurable durations

## API Endpoints

### Core Search

· GET /api/search - Search documentation with advanced query syntax
· GET /api/lookup/id/{entity_id} - Lookup entity by unique ID
· GET /api/lookup/field/{field_id} - Lookup field by unique ID
· GET /api/entity/{entity_name} - Get entity by name

### Management

· POST /api/update - Force refresh documentation data
· GET /api/stats - Get system statistics
· POST /api/cache/clear - Clear memory cache
· GET /health - Health check endpoint

## Advanced Search Syntax

The API supports special query operators for precise searches:

### Type Filtering

Prefix your query with ! followed by the entity type:

```bash
!method sendMessage     # Search only methods named sendMessage
!object User            # Search only objects named User
```

### Property Search

Prefix with . to search within entity properties/parameters:

```bash
.message_id             # Find entities with message_id parameter
.chat_id                # Find entities with chat_id parameter
```

### Wildcard Search

Use * and _ for pattern matching:

```bash
send*                   # Matches sendMessage, sendPhoto, etc.
get_*                   # Matches get_me, get_chat, etc.
chat_*                  # Matches chat_id, chat_member, etc.
```

### Combined Queries

Combine operators for precise results:

```bash
!method .*photo         # Find methods with "photo" in parameters
!object .from           # Find objects with "from" field
```

## Installation

### Prerequisites

· Python 3.10 or higher
· pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/Soumyadeep765/docs-bot.git
cd docs-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py
```

### Docker Deployment

```bash
# Build the image
docker build -t docs-bot .

# Run the container
docker run -p 5000:5000 docs-bot
```

## API Usage Examples

### Basic Search

```bash
curl "http://localhost:5000/api/search?q=sendMessage"
```

### Advanced Search with Type Filter

```bash
curl "http://localhost:5000/api/search?q=!method%20sendMessage&advanced=true"
```

### Lookup by Entity ID

```bash
curl "http://localhost:5000/api/lookup/id/z6ccdr6v"
```

### Lookup by Field ID

```bash
curl "http://localhost:5000/api/lookup/field/z6ccdr6v_1"
```

### Different Output Formats

```bash
# JSON (default)
curl "http://localhost:5000/api/search?q=User&format=normal"

# HTML
curl "http://localhost:5000/api/search?q=User&format=html"

# Markdown
curl "http://localhost:5000/api/search?q=User&format=markdown"
```

### Force Documentation Update

```bash
curl -X POST "http://localhost:5000/api/update"
```

## Response Format

Search Results

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
          "description": "Unique identifier...",
          "required": true
        }
      ],
      "notes": [],
      "reference": "https://core.telegram.org/bots/api#sendmessage",
      "score": 1000
    }
  ],
  "search_time_ms": 0.45,
  "format": "normal",
  "advanced": false
}
```

## Entity Lookup

```json
{
  "id": "a1b2c3d4",
  "name": "sendMessage",
  "type": "method",
  "description": "Sends text messages...",
  "clean_desc": "Sends text messages...",
  "fields": [...],
  "notes": [...],
  "reference": "https://core.telegram.org/bots/api#sendmessage"
}
```

## Configuration

Environment variables can be used to configure the service:

```bash
# Database file location
export DB_FILE=/path/to/database.db

# Cache duration in seconds (default: 3600)
export CACHE_DURATION=1800

# Server port (default: 5000)
export PORT=8080

# Log level (default: INFO)
export LOG_LEVEL=DEBUG
```

## Performance

· In-memory caching: All documentation loaded at startup
· Optimized search: Trie-based keyword matching
· Connection pooling: Async HTTP client with connection reuse
· Database optimizations: WAL mode, memory temp storage
· UVloop integration: High-performance event loop

Typical response times:

· Search queries: < 5ms
· Entity lookups: < 2ms
· Initial load: ~2-3 seconds

## Architecture

### Data Flow

1. Documentation fetched from official Telegram sources
2. HTML parsed and structured into entities
3. Entities assigned unique IDs
4. Data stored in SQLite with full-text search indexes
5. Entire dataset cached in memory for fast access
6. Search queries processed against in-memory cache

### Database Schema

· entities: Main documentation entities
· fields: Entity parameters and properties
· notes: Additional notes and footnotes
· search_index: Full-text search keywords

### Caching Strategy

· Two-level caching: Memory + Database
· LRU eviction policy
· Configurable TTL
· Automatic cache warming on startup

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

· Telegram for the official documentation
· FastAPI team for the excellent web framework
· Contributors and users of this service

## Support

For issues, questions, or feature requests:

· Open an issue on GitHub
· Check existing documentation
· Review the codebase for implementation details

---

Built with performance in mind. Search the Telegram Bot API documentation faster than ever before.