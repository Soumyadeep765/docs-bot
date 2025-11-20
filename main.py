from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
from bs4 import BeautifulSoup
import sqlite3
import re
import httpx
from datetime import datetime, timedelta
import pytz
import asyncio
import logging
from difflib import get_close_matches
from unidecode import unidecode
import os
import json
from typing import List, Optional, Dict, Any
import time
from contextlib import asynccontextmanager
import hashlib
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
BOT_API_URL = 'https://core.telegram.org/bots/api'
WEBAPPS_URL = 'https://core.telegram.org/bots/webapps'
FEATURES_URL = 'https://core.telegram.org/bots/features'
FAQ_URL = 'https://core.telegram.org/bots/faq'
DB_FILE = 'bot_api.db'
CACHE_TTL = 3600  # 1 hour

# Global HTTP client with connection pooling
client = None
# Thread pool for CPU-intensive operations
thread_pool = ThreadPoolExecutor(max_workers=4)
# In-memory cache for frequent queries
memory_cache = {}
cache_lock = threading.Lock()

def adapt_datetime_iso(val):
    return val.isoformat()

def convert_datetime_iso(val):
    return datetime.fromisoformat(val.decode())

sqlite3.register_adapter(datetime, adapt_datetime_iso)
sqlite3.register_converter("timestamp", convert_datetime_iso)

def init_db():
    with sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            content TEXT,
            description TEXT,
            clean_desc TEXT,
            last_updated timestamp NOT NULL,
            source_url TEXT NOT NULL,
            UNIQUE(name, type, source_url)
        )''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS fields (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            type TEXT,
            description TEXT,
            clean_desc TEXT,
            required INTEGER DEFAULT 0,
            FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE,
            UNIQUE(entity_id, name)
        )''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS search_index (
            entity_id INTEGER NOT NULL,
            keyword TEXT NOT NULL,
            weight INTEGER DEFAULT 1,
            FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE,
            PRIMARY KEY (entity_id, keyword)
        )''')
        
        # Enhanced indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entities_type_name ON entities(type, name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entities_name_lower ON entities(LOWER(name))')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_search_keyword_weight ON search_index(keyword, weight)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_fields_entity_name ON fields(entity_id, name)')
        
        cursor.execute('PRAGMA foreign_keys = ON')
        cursor.execute('PRAGMA journal_mode = WAL')
        cursor.execute('PRAGMA synchronous = NORMAL')
        cursor.execute('PRAGMA cache_size = -64000')  # 64MB cache
        cursor.execute('PRAGMA temp_store = MEMORY')
        cursor.execute('PRAGMA mmap_size = 268435456')  # 256MB memory mapping
        conn.commit()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global client
    client = httpx.AsyncClient(
        timeout=30.0,
        limits=httpx.Limits(max_keepalive_connections=50, max_connections=200),
        follow_redirects=True
    )
    
    # Initialize cache
    FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")
    
    # Initialize database
    init_db()
    
    # Load initial data if needed (in background)
    asyncio.create_task(initial_load())
    
    yield
    
    # Shutdown
    if client:
        await client.aclose()
    thread_pool.shutdown(wait=False)

app = FastAPI(title="Telegram Bot API Search", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced cache with faster lookups
class FastSearchCache:
    def __init__(self):
        self._cache = {}
        self._timestamps = {}
        self._hits = 0
        self._misses = 0
    
    def get(self, key):
        with cache_lock:
            if key in self._cache and time.time() - self._timestamps[key] < CACHE_TTL:
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None
    
    def set(self, key, value):
        with cache_lock:
            self._cache[key] = value
            self._timestamps[key] = time.time()
    
    def clear(self):
        with cache_lock:
            self._cache.clear()
            self._timestamps.clear()
    
    def stats(self):
        with cache_lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{hit_rate:.1f}%",
                "size": len(self._cache)
            }

search_cache = FastSearchCache()

# Pre-compiled regex patterns for faster processing
HTML_CLEAN_PATTERNS = {
    'multi_newline': re.compile(r'\n{3,}'),
    'trailing_space': re.compile(r'[ \t]+\n'),
    'leading_space': re.compile(r'\n[ \t]+'),
    'words': re.compile(r'\b\w{3,}\b')
}

async def fetch_url(url):
    try:
        # Cache URL responses to avoid repeated fetches
        cache_key = f"url:{hashlib.md5(url.encode()).hexdigest()}"
        cached = search_cache.get(cache_key)
        if cached:
            return cached
            
        response = await client.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        content_div = soup.find('div', class_='dev_page_content') or soup.find(id='dev_page_content')
        
        if content_div:
            search_cache.set(cache_key, content_div)
        return content_div
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {str(e)}", exc_info=True)
        return None

async def fetch_bot_api():
    content_div = await fetch_url(BOT_API_URL)
    if not content_div:
        return None
    
    return parse_sections(content_div, ['h3', 'h4'])

async def fetch_webapps():
    content_div = await fetch_url(WEBAPPS_URL)
    if not content_div:
        return None
    
    return parse_sections(content_div, ['h2', 'h3', 'h4'])

async def fetch_features():
    content_div = await fetch_url(FEATURES_URL)
    if not content_div:
        return None
    
    return parse_sections(content_div, ['h2', 'h3', 'h4'])

async def fetch_faq():
    content_div = await fetch_url(FAQ_URL)
    if not content_div:
        return None
    
    return parse_sections(content_div, ['h3', 'h4'])

def parse_sections(content_div, heading_tags):
    """Optimized section parsing"""
    sections = []
    current_section = None
    
    for element in content_div.children:
        if element.name in heading_tags:
            if current_section:
                sections.append(current_section)
            current_section = {
                'title': element.get_text().strip(), 
                'level': element.name, 
                'content': '', 
                'elements': [],
                'anchor': element.get('id', '').strip() or element.find_parent('section', {'id': True})
            }
        elif current_section:
            current_section['elements'].append(element)
            current_section['content'] += str(element)
    
    if current_section:
        sections.append(current_section)
    
    return sections

def determine_entity_type(title, source_url):
    if not title:
        return 'other'
    title = title.strip()
    if source_url == BOT_API_URL:
        if title and title[0].islower():
            return 'method'
        if title and title[0].isupper():
            return 'object'
        if 'webapp' in title.lower():
            return 'webapp'
    elif source_url == WEBAPPS_URL:
        return 'webapp'
    elif source_url == FEATURES_URL:
        return 'feature'
    elif source_url == FAQ_URL:
        return 'faq'
    return 'other'

def clean_html_fast(content, source_url):
    """Optimized HTML cleaning"""
    if not content:
        return ""
    soup = BeautifulSoup(content, 'html.parser')
    
    # Remove unwanted tags but keep formatting
    for tag in soup.find_all(True):
        if tag.name not in ['strong', 'b', 'a', 'i', 'em', 'code', 'pre']:
            tag.unwrap()
    
    # Process links
    for a in soup.find_all('a'):
        href = a.get('href', '').strip()
        if href.startswith('#'):
            a['href'] = f"{source_url}{href}"
        elif href.startswith('/'):
            a['href'] = f"https://core.telegram.org{href}"
    
    return str(soup)

def clean_text_fast(content):
    """Optimized text extraction"""
    if not content:
        return ""
    soup = BeautifulSoup(content, 'html.parser')
    return soup.get_text().strip()

def parse_entity_details_fast(content, entity_type, source_url):
    """Optimized entity details parsing"""
    soup = BeautifulSoup(content, 'html.parser')
    details = {'description': '', 'clean_desc': '', 'fields': []}
    description = []
    
    for elem in soup.children:
        if elem.name == 'table':
            break
        description.append(str(elem))
    
    details['description'] = clean_html_fast('\n'.join(description), source_url)
    details['clean_desc'] = clean_text_fast('\n'.join(description))
    
    # Fast table parsing
    for table in soup.find_all('table'):
        headers = [th.text.strip().lower() for th in table.find_all('th')]
        if 'parameter' in headers or 'field' in headers:
            param_index = headers.index('parameter') if 'parameter' in headers else headers.index('field')
            type_index = headers.index('type') if 'type' in headers else -1
            required_index = headers.index('required') if 'required' in headers else -1
            desc_index = headers.index('description') if 'description' in headers else -1
            
            for row in table.find_all('tr')[1:]:
                cols = row.find_all('td')
                if len(cols) <= max(param_index, type_index, required_index, desc_index):
                    continue
                
                field = {
                    'name': cols[param_index].text.strip(),
                    'type': clean_html_fast(str(cols[type_index]), source_url) if type_index != -1 else None,
                    'description': clean_html_fast(str(cols[desc_index]), source_url) if desc_index != -1 else None,
                    'clean_desc': clean_text_fast(str(cols[desc_index])) if desc_index != -1 else None,
                    'required': cols[required_index].text.strip().lower() == 'yes' if required_index != -1 else False
                }
                details['fields'].append(field)
    
    return details

def generate_search_keywords_fast(entity):
    """Optimized keyword generation"""
    keywords = set()
    
    # Name parts
    name_parts = re.findall('[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', entity['name'])
    keywords.update(part.lower() for part in name_parts)
    keywords.add(entity['name'].lower())
    
    # Description keywords
    if entity.get('clean_desc'):
        keywords.update(HTML_CLEAN_PATTERNS['words'].findall(entity['clean_desc'].lower()))
    
    # Field keywords
    for field in entity.get('fields', []):
        keywords.add(field['name'].lower())
        if field.get('type'):
            soup = BeautifulSoup(field['type'], 'html.parser')
            keywords.update(HTML_CLEAN_PATTERNS['words'].findall(soup.get_text().lower()))
    
    # Normalize and filter
    normalized_keywords = set()
    for kw in keywords:
        normalized = unidecode(kw)
        if len(normalized) >= 3:
            normalized_keywords.add(normalized)
    
    return list(normalized_keywords)

def html_to_markdown_fast(html, source_url):
    """Optimized markdown conversion"""
    if not html:
        return ""
    soup = BeautifulSoup(html, 'html.parser')
    
    # Headers
    for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        level = int(h.name[1])
        h.replace_with(f"{'#' * level} {h.get_text().strip()}\n\n")
    
    # Code blocks
    for pre in soup.find_all('pre'):
        code = pre.find('code')
        content = code.get_text().strip() if code else pre.get_text().strip()
        pre.replace_with(f"```\n{content}\n```\n\n")
    
    # Inline code
    for code in soup.find_all('code'):
        if not code.find_parent('pre'):
            code.replace_with(f"`{code.get_text().strip()}`")
    
    # Formatting
    for b in soup.find_all(['b', 'strong']):
        b.replace_with(f"**{b.get_text().strip()}**")
    
    for i in soup.find_all(['i', 'em']):
        i.replace_with(f"*{i.get_text().strip()}*")
    
    # Links
    for a in soup.find_all('a'):
        text = a.get_text().strip()
        href = a.get('href', '').strip()
        if href.startswith('#'):
            href = f"{source_url}{href}"
        elif href.startswith('/'):
            href = f"https://core.telegram.org{href}"
        a.replace_with(f"[{text}]({href})" if href else text)
    
    text = str(soup)
    # Apply regex cleanup
    text = HTML_CLEAN_PATTERNS['multi_newline'].sub('\n\n', text)
    text = HTML_CLEAN_PATTERNS['trailing_space'].sub('\n', text)
    text = HTML_CLEAN_PATTERNS['leading_space'].sub('\n', text)
    
    return text.strip()

def html_to_telegram_mdv1_fast(html, source_url):
    """Optimized Telegram MDv1 conversion"""
    if not html:
        return ""
    soup = BeautifulSoup(html, 'html.parser')
    
    # Headers
    for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        h.replace_with(f"\n*{h.get_text(strip=True)}*\n\n")
    
    # Formatting
    for b in soup.find_all(['b', 'strong']):
        b.replace_with(f"*{b.get_text(strip=True)}*")
    
    for i in soup.find_all(['i', 'em']):
        i.replace_with(f"_{i.get_text(strip=True)}_")
    
    # Code
    for code in soup.find_all('code'):
        if not code.find_parent('pre'):
            code.replace_with(f"`{code.get_text(strip=True)}`")
    
    for pre in soup.find_all('pre'):
        code = pre.get_text().strip()
        pre.replace_with(f"```\n{code}\n```\n\n")
    
    # Links
    for a in soup.find_all('a'):
        text = a.get_text(strip=True)
        href = a.get('href', '').strip()
        if href.startswith('#'):
            href = f"{source_url}{href}"
        elif href.startswith('/'):
            href = f"https://core.telegram.org{href}"
        a.replace_with(f"[{text}]({href})" if href else text)
    
    text = str(soup)
    # Apply regex cleanup
    text = HTML_CLEAN_PATTERNS['multi_newline'].sub('\n\n', text)
    text = HTML_CLEAN_PATTERNS['trailing_space'].sub('\n', text)
    text = HTML_CLEAN_PATTERNS['leading_space'].sub('\n', text)
    
    return text.strip()

def format_content_fast(content, format_type, source_url):
    """Optimized content formatting"""
    if not content:
        return ""
    if format_type == 'html':
        return clean_html_fast(content, source_url)
    elif format_type == 'markdown':
        return html_to_markdown_fast(content, source_url)
    else:  # 'normal' format
        return html_to_telegram_mdv1_fast(content, source_url)

def generate_reference_url(name, source_url, anchor=None):
    if anchor:
        return f"{source_url}#{anchor}"
    name = name.lower().replace(' ', '-')
    if source_url == BOT_API_URL:
        return f"{BOT_API_URL}#{name}"
    elif source_url == WEBAPPS_URL:
        return f"{WEBAPPS_URL}#{name}"
    elif source_url == FEATURES_URL:
        return f"{FEATURES_URL}#{name}"
    elif source_url == FAQ_URL:
        return f"{FAQ_URL}#{name}"
    return source_url

async def update_docs():
    """Optimized documentation update"""
    now = datetime.now(pytz.utc)
    
    with sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
        cursor = conn.cursor()
        
        # Clear old data
        cursor.execute("DELETE FROM entities WHERE source_url IN (?, ?, ?, ?)", 
                      (BOT_API_URL, WEBAPPS_URL, FEATURES_URL, FAQ_URL))
        cursor.execute("DELETE FROM fields WHERE entity_id NOT IN (SELECT id FROM entities)")
        cursor.execute("DELETE FROM search_index WHERE entity_id NOT IN (SELECT id FROM entities)")
        
        # Fetch all sources concurrently
        tasks = [
            fetch_bot_api(),
            fetch_webapps(),
            fetch_features(),
            fetch_faq()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        sources = [
            (BOT_API_URL, results[0] if not isinstance(results[0], Exception) else []),
            (WEBAPPS_URL, results[1] if not isinstance(results[1], Exception) else []),
            (FEATURES_URL, results[2] if not isinstance(results[2], Exception) else []),
            (FAQ_URL, results[3] if not isinstance(results[3], Exception) else [])
        ]
        
        entity_data = []
        field_data = []
        keyword_data = []
        
        for source_url, sections in sources:
            if not sections:
                continue
                
            for section in sections:
                entity_type = determine_entity_type(section['title'], source_url)
                if entity_type == 'other':
                    continue
                    
                details = parse_entity_details_fast(section['content'], entity_type, source_url)
                
                # Collect entity data for batch insert
                entity_data.append((
                    section['title'], 
                    entity_type, 
                    section['content'], 
                    details['description'], 
                    details['clean_desc'], 
                    now, 
                    source_url
                ))
        
        # Batch insert entities
        if entity_data:
            cursor.executemany(
                '''INSERT INTO entities (name, type, content, description, clean_desc, last_updated, source_url)
                VALUES (?, ?, ?, ?, ?, ?, ?)''',
                entity_data
            )
            
            # Get all inserted entity IDs
            cursor.execute("SELECT id, name, clean_desc FROM entities WHERE last_updated = ?", (now,))
            new_entities = cursor.fetchall()
            
            # Prepare field and keyword data
            for entity_id, name, clean_desc in new_entities:
                # Find matching details
                for entity_row in entity_data:
                    if entity_row[0] == name:  # Match by name
                        details = parse_entity_details_fast(entity_row[2], entity_row[1], entity_row[6])
                        
                        # Fields
                        for field in details.get('fields', []):
                            field_data.append((
                                entity_id, 
                                field['name'], 
                                field.get('type'), 
                                field.get('description'), 
                                field.get('clean_desc'), 
                                field.get('required', False)
                            ))
                        
                        # Keywords
                        keywords = generate_search_keywords_fast({
                            'name': name,
                            'clean_desc': clean_desc,
                            'fields': details.get('fields', [])
                        })
                        
                        for keyword in keywords:
                            weight = 1
                            if any(f['name'].lower() == keyword for f in details.get('fields', [])):
                                weight = 2
                            if name.lower() == keyword:
                                weight = 3
                            keyword_data.append((entity_id, keyword, weight))
                        break
        
        # Batch insert fields and keywords
        if field_data:
            cursor.executemany(
                '''INSERT INTO fields (entity_id, name, type, description, clean_desc, required)
                VALUES (?, ?, ?, ?, ?, ?)''',
                field_data
            )
        
        if keyword_data:
            cursor.executemany(
                "INSERT INTO search_index (entity_id, keyword, weight) VALUES (?, ?, ?)",
                keyword_data
            )
        
        conn.commit()
    
    # Clear search cache after update
    search_cache.clear()
    
    return True

async def initial_load():
    """Optimized initial load"""
    db_exists = os.path.exists(DB_FILE)
    if db_exists:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT 1 FROM entities LIMIT 1')
            has_data = cursor.fetchone() is not None
    else:
        has_data = False
        
    if not db_exists or not has_data:
        logger.info("Initial database load required")
        success = await update_docs()
        if not success:
            logger.error("Initial database load failed")
    else:
        logger.info("Database already exists with data")

@app.get("/api/search")
@cache(expire=300)  # Cache for 5 minutes
async def search(
    q: str = Query(..., description="Search query"),
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    limit: int = Query(20, ge=1, le=100, description="Number of results"),
    format: str = Query("normal", description="Output format: normal, markdown, or html")
):
    """Optimized search endpoint"""
    if format not in ['normal', 'markdown', 'html']:
        raise HTTPException(status_code=400, detail="Invalid format parameter. Use 'normal', 'markdown' or 'html'")
    
    query = q.strip().lower()
    if not query:
        raise HTTPException(status_code=400, detail="Missing search query")
    
    # Check cache first
    cache_key = f"search:{query}:{entity_type}:{limit}:{format}"
    cached_result = search_cache.get(cache_key)
    if cached_result:
        return cached_result
    
    try:
        with sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Use a single optimized query with UNION for better performance
            search_query = '''
            WITH ranked_results AS (
                -- Exact matches (highest priority)
                SELECT e.id, e.name, e.type, e.description, e.clean_desc, 
                       e.content, e.source_url, 1000 as priority
                FROM entities e
                WHERE LOWER(e.name) = ?
                {type_filter}
                
                UNION ALL
                
                -- Keyword matches
                SELECT e.id, e.name, e.type, e.description, e.clean_desc, 
                       e.content, e.source_url, SUM(si.weight) as priority
                FROM entities e
                JOIN search_index si ON e.id = si.entity_id
                WHERE si.keyword LIKE ? || '%'
                {type_filter}
                GROUP BY e.id
                
                UNION ALL
                
                -- Similar matches (fuzzy)
                SELECT e.id, e.name, e.type, e.description, e.clean_desc, 
                       e.content, e.source_url, 
                       CASE WHEN LOWER(e.name) LIKE ? || '%' THEN 10 ELSE 5 END as priority
                FROM entities e
                WHERE (LOWER(e.name) LIKE ? || '%' OR e.clean_desc LIKE ?)
                {type_filter}
            )
            SELECT DISTINCT id, name, type, description, clean_desc, content, source_url, priority
            FROM ranked_results
            ORDER BY priority DESC, name
            LIMIT ?
            '''
            
            # Build type filter
            type_filter = "AND e.type = ?" if entity_type else ""
            search_query = search_query.format(type_filter=type_filter)
            
            # Prepare parameters
            params = [query]
            if entity_type:
                params.append(entity_type)
            
            params.extend([query, query, f"%{query}%", query, limit])
            
            cursor.execute(search_query, params)
            results_data = [dict(row) for row in cursor.fetchall()]
            
            # Batch fetch fields for all results
            if results_data:
                entity_ids = [str(r['id']) for r in results_data]
                placeholders = ','.join(['?'] * len(entity_ids))
                
                cursor.execute(f'''
                SELECT entity_id, name, type, description, clean_desc, required
                FROM fields
                WHERE entity_id IN ({placeholders})
                ORDER BY entity_id, required DESC, name
                ''', entity_ids)
                
                fields_data = {}
                for row in cursor.fetchall():
                    entity_id = row[0]
                    if entity_id not in fields_data:
                        fields_data[entity_id] = []
                    fields_data[entity_id].append(dict(row))
            
            # Process results
            results = []
            for result in results_data:
                entity_id = result['id']
                fields = fields_data.get(entity_id, [])
                
                formatted_fields = []
                for field in fields:
                    formatted_fields.append({
                        'name': field['name'],
                        'type': format_content_fast(field['type'], format, result['source_url']),
                        'description': format_content_fast(field['description'], format, result['source_url']),
                        'clean_desc': field['clean_desc'],
                        'required': field['required']
                    })
                
                results.append({
                    'id': entity_id,
                    'name': result['name'],
                    'type': result['type'],
                    'description': format_content_fast(result.get('description'), format, result['source_url']),
                    'clean_desc': result.get('clean_desc'),
                    'content': format_content_fast(result.get('content'), format, result['source_url']),
                    'fields': formatted_fields,
                    'reference': generate_reference_url(result['name'], result['source_url']),
                    'match_type': 'exact' if result['priority'] == 1000 else 'keyword' if result['priority'] >= 10 else 'similar'
                })
            
            response = {
                'query': query,
                'count': len(results),
                'results': results,
                'format': format
            }
            
            # Cache the response
            search_cache.set(cache_key, response)
            
            return response
            
    except Exception as e:
        logger.error(f"Search failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/types")
@cache(expire=3600)
async def list_types():
    return {
        "types": [
            {"id": "method", "name": "Methods"},
            {"id": "object", "name": "Objects"},
            {"id": "webapp", "name": "WebApp Features"},
            {"id": "feature", "name": "Bot Features"},
            {"id": "faq", "name": "FAQ"},
            {"id": "other", "name": "Other"}
        ]
    }

@app.get("/api/list")
@cache(expire=300)
async def list_entities(entity_type: str = Query(..., description="Entity type")):
    valid_types = ['method', 'object', 'webapp', 'feature', 'faq', 'other']
    if entity_type not in valid_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid type. Valid types are: {', '.join(valid_types)}"
        )
    
    try:
        with sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT name
                FROM entities 
                WHERE type = ?
                ORDER BY name
            ''', (entity_type,))
            
            names = [row[0] for row in cursor.fetchall()]
            return names
            
    except Exception as e:
        logger.error(f"List entities failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/update")
async def update_docs_endpoint(background_tasks: BackgroundTasks):
    background_tasks.add_task(update_docs)
    return {"status": "update_started", "message": "Documentation update started in background"}

@app.get("/cache/stats")
async def cache_stats():
    return search_cache.stats()

@app.get("/")
async def root():
    return {"message": "Telegram Bot API Search Service", "status": "running", "optimized": True}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(pytz.utc).isoformat()}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        workers=1,
        loop="asyncio"
    )