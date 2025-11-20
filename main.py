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
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS custom_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_type TEXT NOT NULL,
            name TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata TEXT,
            last_updated timestamp NOT NULL,
            UNIQUE(data_type, name)
        )''')
        
        # Add indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_search_keyword ON search_index(keyword)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_fields_entity ON fields(entity_id)')
        
        cursor.execute('PRAGMA foreign_keys = ON')
        cursor.execute('PRAGMA journal_mode = WAL')
        cursor.execute('PRAGMA synchronous = NORMAL')
        cursor.execute('PRAGMA cache_size = -10000')  # 10MB cache
        conn.commit()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global client
    client = httpx.AsyncClient(
        timeout=30.0,
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        http2=True
    )
    
    # Initialize cache
    FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")
    
    # Initialize database
    init_db()
    
    # Load initial data if needed
    await initial_load()
    
    yield
    
    # Shutdown
    if client:
        await client.aclose()

app = FastAPI(title="Telegram Bot API Search", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache for expensive operations
class SearchCache:
    def __init__(self):
        self._cache = {}
        self._timestamps = {}
    
    def get(self, key):
        if key in self._cache and time.time() - self._timestamps[key] < CACHE_TTL:
            return self._cache[key]
        return None
    
    def set(self, key, value):
        self._cache[key] = value
        self._timestamps[key] = time.time()
    
    def clear(self):
        self._cache.clear()
        self._timestamps.clear()

search_cache = SearchCache()

async def fetch_url(url):
    try:
        response = await client.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        content_div = soup.find('div', class_='dev_page_content') or soup.find(id='dev_page_content')
        return content_div
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {str(e)}", exc_info=True)
        return None

async def fetch_bot_api():
    content_div = await fetch_url(BOT_API_URL)
    if not content_div:
        return None
    
    sections = []
    current_section = None
    
    for element in content_div.children:
        if element.name in ['h3', 'h4']:
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

async def fetch_webapps():
    content_div = await fetch_url(WEBAPPS_URL)
    if not content_div:
        return None
    
    sections = []
    current_section = None
    
    for element in content_div.children:
        if element.name in ['h2', 'h3', 'h4']:
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

async def fetch_features():
    content_div = await fetch_url(FEATURES_URL)
    if not content_div:
        return None
    
    sections = []
    current_section = None
    
    for element in content_div.children:
        if element.name in ['h2', 'h3', 'h4']:
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

async def fetch_faq():
    content_div = await fetch_url(FAQ_URL)
    if not content_div:
        return None
    
    sections = []
    current_section = None
    
    for element in content_div.children:
        if element.name in ['h3', 'h4']:
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

def clean_html(content, source_url):
    if not content:
        return ""
    soup = BeautifulSoup(content, 'html.parser')
    for tag in soup.find_all(True):
        if tag.name not in ['strong', 'b', 'a', 'i', 'em', 'code', 'pre']:
            tag.unwrap()
    for a in soup.find_all('a'):
        href = a.get('href', '').strip()
        if href.startswith('#'):
            a['href'] = f"{source_url}{href}"
        elif href.startswith('/'):
            a['href'] = f"https://core.telegram.org{href}"
    return str(soup)

def clean_text(content):
    if not content:
        return ""
    soup = BeautifulSoup(content, 'html.parser')
    return soup.get_text().strip()

def parse_entity_details(content, entity_type, source_url):
    soup = BeautifulSoup(content, 'html.parser')
    details = {'description': '', 'clean_desc': '', 'fields': []}
    description = []
    
    for elem in soup.children:
        if elem.name == 'table':
            break
        description.append(str(elem))
    
    details['description'] = clean_html('\n'.join(description), source_url)
    details['clean_desc'] = clean_text('\n'.join(description))
    
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
                    'type': clean_html(str(cols[type_index]), source_url) if type_index != -1 else None,
                    'description': clean_html(str(cols[desc_index]), source_url) if desc_index != -1 else None,
                    'clean_desc': clean_text(str(cols[desc_index])) if desc_index != -1 else None,
                    'required': cols[required_index].text.strip().lower() == 'yes' if required_index != -1 else False
                }
                details['fields'].append(field)
    
    return details

def generate_search_keywords(entity):
    keywords = set()
    name_parts = re.findall('[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', entity['name'])
    keywords.update(part.lower() for part in name_parts)
    keywords.add(entity['name'].lower())
    
    if entity.get('clean_desc'):
        keywords.update(re.findall(r'\b\w{3,}\b', entity['clean_desc'].lower()))
    
    for field in entity.get('fields', []):
        keywords.add(field['name'].lower())
        if field.get('type'):
            soup = BeautifulSoup(field['type'], 'html.parser')
            keywords.update(re.findall(r'\b\w{3,}\b', soup.get_text().lower()))
    
    normalized_keywords = set()
    for kw in keywords:
        normalized = unidecode(kw)
        if len(normalized) >= 3:
            normalized_keywords.add(normalized)
    
    return list(normalized_keywords)

def html_to_markdown(html, source_url):
    if not html:
        return ""
    soup = BeautifulSoup(html, 'html.parser')
    
    for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        level = int(h.name[1])
        h.replace_with(f"{'#' * level} {h.get_text().strip()}\n\n")
    
    for pre in soup.find_all('pre'):
        code = pre.find('code')
        if code:
            pre.replace_with(f"```\n{code.get_text().strip()}\n```\n\n")
        else:
            pre.replace_with(f"```\n{pre.get_text().strip()}\n```\n\n")
    
    for code in soup.find_all('code'):
        if not code.find_parent('pre'):
            code.replace_with(f"`{code.get_text().strip()}`")
    
    for b in soup.find_all(['b', 'strong']):
        b.replace_with(f"**{b.get_text().strip()}**")
    
    for i in soup.find_all(['i', 'em']):
        i.replace_with(f"*{i.get_text().strip()}*")
    
    for a in soup.find_all('a'):
        text = a.get_text().strip()
        href = a.get('href', '').strip()
        if href.startswith('#'):
            href = f"{source_url}{href}"
        elif href.startswith('/'):
            href = f"https://core.telegram.org{href}"
        a.replace_with(f"[{text}]({href})" if href else text)
    
    text = str(soup)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n[ \t]+', '\n', text)
    return text.strip()

def html_to_telegram_mdv1(html, source_url):
    if not html:
        return ""
    soup = BeautifulSoup(html, 'html.parser')
    
    for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        h.replace_with(f"\n*{h.get_text(strip=True)}*\n\n")
    
    for b in soup.find_all(['b', 'strong']):
        b.replace_with(f"*{b.get_text(strip=True)}*")
    
    for i in soup.find_all(['i', 'em']):
        i.replace_with(f"_{i.get_text(strip=True)}_")
    
    for code in soup.find_all('code'):
        if not code.find_parent('pre'):
            code.replace_with(f"`{code.get_text(strip=True)}`")
    
    for pre in soup.find_all('pre'):
        code = pre.get_text().strip()
        pre.replace_with(f"```\n{code}\n```\n\n")
    
    for a in soup.find_all('a'):
        text = a.get_text(strip=True)
        href = a.get('href', '').strip()
        if href.startswith('#'):
            href = f"{source_url}{href}"
        elif href.startswith('/'):
            href = f"https://core.telegram.org{href}"
        a.replace_with(f"[{text}]({href})" if href else text)
    
    text = str(soup)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n[ \t]+', '\n', text)
    return text.strip()

def format_content(content, format_type, source_url):
    if not content:
        return ""
    if format_type == 'html':
        return clean_html(content, source_url)
    elif format_type == 'markdown':
        return html_to_markdown(content, source_url)
    else:
        return html_to_telegram_mdv1(content, source_url)

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
        results = await asyncio.gather(*tasks)
        
        sources = [
            (BOT_API_URL, results[0]),
            (WEBAPPS_URL, results[1]),
            (FEATURES_URL, results[2]),
            (FAQ_URL, results[3])
        ]
        
        for source_url, sections in sources:
            if not sections:
                continue
                
            for section in sections:
                entity_type = determine_entity_type(section['title'], source_url)
                if entity_type == 'other':
                    continue
                    
                details = parse_entity_details(section['content'], entity_type, source_url)
                
                cursor.execute(
                    '''INSERT INTO entities (name, type, content, description, clean_desc, last_updated, source_url)
                    VALUES (?, ?, ?, ?, ?, ?, ?)''',
                    (
                        section['title'], 
                        entity_type, 
                        section['content'], 
                        details['description'], 
                        details['clean_desc'], 
                        now, 
                        source_url
                    )
                )
                entity_id = cursor.lastrowid
                
                # Batch insert fields
                field_data = []
                for field in details.get('fields', []):
                    field_data.append((
                        entity_id, 
                        field['name'], 
                        field.get('type'), 
                        field.get('description'), 
                        field.get('clean_desc'), 
                        field.get('required', False)
                    ))
                
                if field_data:
                    cursor.executemany(
                        '''INSERT INTO fields (entity_id, name, type, description, clean_desc, required)
                        VALUES (?, ?, ?, ?, ?, ?)''',
                        field_data
                    )
                
                # Generate and insert keywords
                keywords = generate_search_keywords({
                    'name': section['title'],
                    'clean_desc': details['clean_desc'],
                    'fields': details.get('fields', [])
                })
                
                keyword_data = []
                for keyword in keywords:
                    weight = 1
                    if any(f['name'].lower() == keyword for f in details.get('fields', [])):
                        weight = 2
                    if section['title'].lower() == keyword:
                        weight = 3
                    keyword_data.append((entity_id, keyword, weight))
                
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
    format_type: str = Query("normal", description="Output format: normal, markdown, or html")
):
    if format_type not in ['normal', 'markdown', 'html']:
        raise HTTPException(status_code=400, detail="Invalid format parameter. Use 'normal', 'markdown' or 'html'")
    
    query = q.strip().lower()
    if not query:
        raise HTTPException(status_code=400, detail="Missing search query")
    
    # Check cache first
    cache_key = f"search:{query}:{entity_type}:{limit}:{format_type}"
    cached_result = search_cache.get(cache_key)
    if cached_result:
        return cached_result
    
    try:
        with sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Exact match query
            exact_query = '''
            SELECT e.id, e.name, e.type, e.description, e.clean_desc, 
                   COALESCE(e.content, '') as content, e.source_url
            FROM entities e
            WHERE LOWER(e.name) = ?
            '''
            exact_params = [query]
            if entity_type:
                exact_query += ' AND e.type = ?'
                exact_params.append(entity_type)
            
            cursor.execute(exact_query, exact_params)
            exact_matches = [dict(row) for row in cursor.fetchall()]
            
            # Keyword match query
            keyword_query = '''
            SELECT e.id, e.name, e.type, e.description, e.clean_desc, 
                   COALESCE(e.content, '') as content, SUM(si.weight) as score, e.source_url
            FROM entities e
            JOIN search_index si ON e.id = si.entity_id
            WHERE si.keyword LIKE ? || '%'
            '''
            keyword_params = [query]
            if entity_type:
                keyword_query += ' AND e.type = ?'
                keyword_params.append(entity_type)
            
            keyword_query += '''
            GROUP BY e.id
            ORDER BY score DESC, e.name
            LIMIT ?
            '''
            keyword_params.append(limit)
            
            cursor.execute(keyword_query, keyword_params)
            keyword_matches = [dict(row) for row in cursor.fetchall()]
            
            # Similar matches using cached entity list
            cache_key_entities = f"entities:{entity_type}"
            all_entities = search_cache.get(cache_key_entities)
            
            if all_entities is None:
                if entity_type:
                    cursor.execute('SELECT name, type FROM entities WHERE type = ?', (entity_type,))
                else:
                    cursor.execute('SELECT name, type FROM entities')
                all_entities = [(row[0].lower(), row[1]) for row in cursor.fetchall()]
                search_cache.set(cache_key_entities, all_entities)
            
            if entity_type:
                all_names = [name for name, typ in all_entities if typ == entity_type]
            else:
                all_names = [name for name, typ in all_entities]
            
            similar_names = get_close_matches(query, all_names, n=5, cutoff=0.3)
            
            similar_results = []
            if similar_names:
                placeholders = ','.join(['?'] * len(similar_names))
                similar_query = f'''
                SELECT e.id, e.name, e.type, e.description, e.clean_desc, 
                       COALESCE(e.content, '') as content, e.source_url
                FROM entities e
                WHERE LOWER(e.name) IN ({placeholders})
                '''
                similar_params = similar_names
                if entity_type:
                    similar_query += ' AND e.type = ?'
                    similar_params.append(entity_type)
                
                cursor.execute(similar_query, similar_params)
                similar_results = [dict(row) for row in cursor.fetchall()]
            
            # Combine and deduplicate results
            combined_results = []
            seen_ids = set()
            
            for match in exact_matches + keyword_matches + similar_results:
                if match['id'] not in seen_ids:
                    seen_ids.add(match['id'])
                    combined_results.append(match)
            
            # Process results
            results = []
            for result in combined_results[:limit]:
                # Batch fetch fields for all results
                cursor.execute('''
                SELECT name, type, description, clean_desc, required
                FROM fields
                WHERE entity_id = ?
                ORDER BY required DESC, name
                ''', (result['id'],))
                fields = [dict(row) for row in cursor.fetchall()]
                
                formatted_fields = []
                for field in fields:
                    formatted_fields.append({
                        'name': field['name'],
                        'type': format_content(field['type'], format_type, result['source_url']),
                        'description': format_content(field['description'], format_type, result['source_url']),
                        'clean_desc': field['clean_desc'],
                        'required': field['required']
                    })
                
                results.append({
                    'id': result['id'],
                    'name': result['name'],
                    'type': result['type'],
                    'description': format_content(result.get('description'), format_type, result['source_url']),
                    'clean_desc': result.get('clean_desc'),
                    'content': format_content(result.get('content'), format_type, result['source_url']),
                    'fields': formatted_fields,
                    'reference': generate_reference_url(result['name'], result['source_url']),
                    'match_type': 'exact' if result in exact_matches else 
                                'keyword' if result in keyword_matches else 
                                'similar'
                })
            
            response = {
                'query': query,
                'count': len(results),
                'results': results,
                'format': format_type
            }
            
            # Cache the response
            search_cache.set(cache_key, response)
            
            return response
            
    except Exception as e:
        logger.error(f"Search failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/types")
@cache(expire=3600)  # Cache for 1 hour
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
@cache(expire=300)  # Cache for 5 minutes
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

@app.get("/")
async def root():
    return {"message": "Telegram Bot API Search Service", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(pytz.utc).isoformat()}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        workers=1,  # Single worker for SQLite compatibility
        loop="asyncio"
    )