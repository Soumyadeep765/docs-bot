from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sqlite3
import asyncio
import aiosqlite
import httpx
import orjson
from datetime import datetime, timedelta
import pytz
import logging
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import pickle
import os
import re
from contextlib import asynccontextmanager
import uvloop
from bs4 import BeautifulSoup
from unidecode import unidecode

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BOT_API_URL = 'https://core.telegram.org/bots/api'
WEBAPPS_URL = 'https://core.telegram.org/bots/webapps'
FEATURES_URL = 'https://core.telegram.org/bots/features'
FAQ_URL = 'https://core.telegram.org/bots/faq'
DB_FILE = 'bot_api.db'
CACHE_DURATION = 3600

class MemoryCache:
    def __init__(self):
        self._cache = {}
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str):
        if key in self._cache:
            data, expiry = self._cache[key]
            if expiry > datetime.now().timestamp():
                self._hits += 1
                return data
            else:
                del self._cache[key]
        self._misses += 1
        return None
    
    def set(self, key: str, data: Any, duration: int = CACHE_DURATION):
        expiry = datetime.now().timestamp() + duration
        self._cache[key] = (data, expiry)
    
    def clear(self):
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    def stats(self):
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "size": len(self._cache)
        }

memory_cache = MemoryCache()

CLEAN_PATTERN = re.compile(r'<.*?>')
WORD_PATTERN = re.compile(r'\b\w{3,}\b')
NAME_PATTERN = re.compile(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)')
NOTE_PATTERN = re.compile(r'<strong>(\d+)\.</strong>\s*(.*?)(?=<strong>\d+\.</strong>|</blockquote>|$)', re.DOTALL)
BLOCKQUOTE_PATTERN = re.compile(r'<blockquote>(.*?)</blockquote>', re.DOTALL)

full_db_cache = {}

client = httpx.AsyncClient(
    timeout=30.0,
    limits=httpx.Limits(max_keepalive_connections=50, max_connections=200),
    http2=True
)

async def init_db():
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute('PRAGMA journal_mode=WAL')
        await db.execute('PRAGMA synchronous=NORMAL')
        await db.execute('PRAGMA cache_size=-200000')
        await db.execute('PRAGMA temp_store=memory')
        
        await db.execute('''
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            content TEXT,
            description_html TEXT,
            description_text TEXT,
            description_markdown TEXT,
            clean_desc TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            source_url TEXT NOT NULL,
            UNIQUE(name, type, source_url)
        )
        ''')
        
        await db.execute('''
        CREATE TABLE IF NOT EXISTS fields (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            type TEXT,
            description_html TEXT,
            description_text TEXT,
            description_markdown TEXT,
            clean_desc TEXT,
            required INTEGER DEFAULT 0,
            FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE,
            UNIQUE(entity_id, name)
        )
        ''')
        
        await db.execute('''
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER NOT NULL,
            note_number INTEGER NOT NULL,
            content_html TEXT NOT NULL,
            content_text TEXT NOT NULL,
            content_markdown TEXT NOT NULL,
            clean_desc TEXT NOT NULL,
            FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE,
            UNIQUE(entity_id, note_number)
        )
        ''')
        
        await db.execute('''
        CREATE TABLE IF NOT EXISTS search_index (
            entity_id INTEGER NOT NULL,
            keyword TEXT NOT NULL,
            weight INTEGER DEFAULT 1,
            FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE,
            PRIMARY KEY (entity_id, keyword)
        )
        ''')
        
        await db.execute('CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)')
        await db.execute('CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type)')
        await db.execute('CREATE INDEX IF NOT EXISTS idx_search_keyword ON search_index(keyword)')
        await db.execute('CREATE INDEX IF NOT EXISTS idx_search_entity ON search_index(entity_id)')
        await db.execute('CREATE INDEX IF NOT EXISTS idx_notes_entity ON notes(entity_id)')
        
        await db.execute('PRAGMA foreign_keys = ON')
        await db.commit()
        
        logger.info("Database initialized with optimized schema including notes")

async def load_full_db_cache():
    global full_db_cache
    
    async with aiosqlite.connect(DB_FILE) as db:
        db.row_factory = aiosqlite.Row
        
        entities = {}
        async with db.execute('''
            SELECT id, name, type, description_html, description_text, description_markdown, 
                   clean_desc, source_url
            FROM entities
        ''') as cursor:
            async for row in cursor:
                entities[row['id']] = dict(row)
        
        search_terms = {}
        async with db.execute('''
            SELECT entity_id, keyword, weight 
            FROM search_index
        ''') as cursor:
            async for row in cursor:
                entity_id = row['entity_id']
                if entity_id not in search_terms:
                    search_terms[entity_id] = []
                search_terms[entity_id].append((row['keyword'], row['weight']))
        
        entity_fields = {}
        async with db.execute('''
            SELECT entity_id, name, type, description_html, description_text, description_markdown, clean_desc, required
            FROM fields
        ''') as cursor:
            async for row in cursor:
                entity_id = row['entity_id']
                if entity_id not in entity_fields:
                    entity_fields[entity_id] = []
                entity_fields[entity_id].append(dict(row))
        
        entity_notes = {}
        async with db.execute('''
            SELECT entity_id, note_number, content_html, content_text, content_markdown, clean_desc
            FROM notes
            ORDER BY note_number
        ''') as cursor:
            async for row in cursor:
                entity_id = row['entity_id']
                if entity_id not in entity_notes:
                    entity_notes[entity_id] = []
                entity_notes[entity_id].append({
                    'number': row['note_number'],
                    'html': row['content_html'],
                    'text': row['content_text'],
                    'markdown': row['content_markdown'],
                    'clean_desc': row['clean_desc']
                })
        
        full_db_cache = {
            'entities': entities,
            'search_terms': search_terms,
            'entity_fields': entity_fields,
            'entity_notes': entity_notes,
            'last_updated': datetime.now(pytz.utc),
            'entity_count': len(entities),
            'term_count': sum(len(terms) for terms in search_terms.values())
        }
    
    logger.info(f"📦 Full DB cache loaded: {full_db_cache['entity_count']} entities, {full_db_cache['term_count']} terms")

async def initial_data_load():
    async with aiosqlite.connect(DB_FILE) as db:
        cursor = await db.execute('SELECT COUNT(*) FROM entities')
        count = (await cursor.fetchone())[0]
        
        if count == 0:
            logger.info("No data found, performing initial data load...")
            await update_documentation_data()
        else:
            logger.info(f"Database contains {count} entities")

async def update_documentation_data():
    try:
        logger.info("Starting documentation update...")
        
        async with aiosqlite.connect(DB_FILE) as db:
            await db.execute('DELETE FROM entities')
            await db.execute('DELETE FROM fields')
            await db.execute('DELETE FROM notes')
            await db.execute('DELETE FROM search_index')
            await db.commit()
        
        sources = [
            (BOT_API_URL, await fetch_bot_api()),
            (WEBAPPS_URL, await fetch_webapps()),
            (FEATURES_URL, await fetch_features()),
            (FAQ_URL, await fetch_faq())
        ]
        
        entity_count = 0
        for source_url, sections in sources:
            if not sections:
                continue
                
            for section in sections:
                await save_entity_to_db(section, source_url)
                entity_count += 1
        
        logger.info(f"Documentation update completed: {entity_count} entities")
        return True
        
    except Exception as e:
        logger.error(f"Documentation update failed: {e}")
        return False

async def fetch_bot_api():
    try:
        response = await client.get(BOT_API_URL)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        content_div = soup.find('div', class_='dev_page_content') or soup.find(id='dev_page_content')
        return parse_sections(content_div, ['h3', 'h4']) if content_div else None
    except Exception as e:
        logger.error(f"Failed to fetch Bot API: {e}")
        return None

async def fetch_webapps():
    try:
        response = await client.get(WEBAPPS_URL)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        content_div = soup.find('div', class_='dev_page_content') or soup.find(id='dev_page_content')
        return parse_sections(content_div, ['h2', 'h3', 'h4']) if content_div else None
    except Exception as e:
        logger.error(f"Failed to fetch WebApps: {e}")
        return None

async def fetch_features():
    try:
        response = await client.get(FEATURES_URL)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        content_div = soup.find('div', class_='dev_page_content') or soup.find(id='dev_page_content')
        return parse_sections(content_div, ['h2', 'h3', 'h4']) if content_div else None
    except Exception as e:
        logger.error(f"Failed to fetch Features: {e}")
        return None

async def fetch_faq():
    try:
        response = await client.get(FAQ_URL)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        content_div = soup.find('div', class_='dev_page_content') or soup.find(id='dev_page_content')
        return parse_sections(content_div, ['h3', 'h4']) if content_div else None
    except Exception as e:
        logger.error(f"Failed to fetch FAQ: {e}")
        return None

def parse_sections(content_div, heading_tags):
    if not content_div:
        return []
        
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
                'anchor': element.get('id', '')
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

def extract_notes(content: str) -> Tuple[str, List[Dict]]:
    if not content:
        return content, []
    
    notes = []
    
    blockquote_matches = BLOCKQUOTE_PATTERN.findall(content)
    
    for blockquote in blockquote_matches:
        note_matches = NOTE_PATTERN.findall(blockquote)
        
        for note_num, note_content in note_matches:
            note_num = int(note_num.strip())
            note_content = note_content.strip()
            
            html_content = f"<strong>{note_num}.</strong> {note_content}"
            text_content = clean_text(html_content)
            markdown_content = html_to_markdown(html_content)
            clean_desc = clean_text(note_content)
            
            notes.append({
                'number': note_num,
                'html': html_content,
                'text': text_content,
                'markdown': markdown_content,
                'clean_desc': clean_desc
            })
    
    cleaned_content = BLOCKQUOTE_PATTERN.sub('', content)
    
    return cleaned_content, notes

async def save_entity_to_db(section, source_url):
    entity_type = determine_entity_type(section['title'], source_url)
    if entity_type == 'other':
        return
    
    if not section.get('content') and section.get('elements'):
        section['content'] = ''.join(str(elem) for elem in section['elements'])
    
    cleaned_content, notes = extract_notes(section['content'])
    
    details = parse_entity_details(cleaned_content, entity_type, source_url)
    
    async with aiosqlite.connect(DB_FILE) as db:
        cursor = await db.execute(
            '''INSERT INTO entities (name, type, content, description_html, description_text, 
            description_markdown, clean_desc, last_updated, source_url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                section['title'],
                entity_type,
                section['content'],
                details['description_html'],
                details['description_text'],
                details['description_markdown'],
                details['clean_desc'],
                datetime.now(pytz.utc),
                source_url
            )
        )
        entity_id = cursor.lastrowid
        
        for note in notes:
            await db.execute(
                '''INSERT INTO notes (entity_id, note_number, content_html, content_text, content_markdown, clean_desc)
                VALUES (?, ?, ?, ?, ?, ?)''',
                (
                    entity_id,
                    note['number'],
                    note['html'],
                    note['text'],
                    note['markdown'],
                    note['clean_desc']
                )
            )
        
        for field in details.get('fields', []):
            await db.execute(
                '''INSERT INTO fields (entity_id, name, type, description_html, description_text, 
                description_markdown, clean_desc, required)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    entity_id,
                    field['name'],
                    field.get('type'),
                    field.get('description_html'),
                    field.get('description_text'),
                    field.get('description_markdown'),
                    field.get('clean_desc'),
                    field.get('required', False)
                )
            )
        
        keywords = generate_search_keywords({
            'name': section['title'],
            'clean_desc': details['clean_desc'],
            'fields': details.get('fields', [])
        })
        
        for keyword in keywords:
            weight = 1
            if any(f['name'].lower() == keyword for f in details.get('fields', [])):
                weight = 2
            if section['title'].lower() == keyword:
                weight = 3
                
            await db.execute(
                "INSERT INTO search_index (entity_id, keyword, weight) VALUES (?, ?, ?)",
                (entity_id, keyword, weight)
            )
        
        await db.commit()

def parse_entity_details(content, entity_type, source_url):
    if not content:
        return {
            'description_html': '',
            'description_text': '',
            'description_markdown': '',
            'clean_desc': '',
            'fields': []
        }
    
    soup = BeautifulSoup(content, 'html.parser')
    details = {
        'description_html': '',
        'description_text': '',
        'description_markdown': '',
        'clean_desc': '',
        'fields': []
    }
    
    description_elements = []
    for elem in soup.children:
        if elem.name == 'table':
            break
        description_elements.append(str(elem))
    
    description_html = clean_html(''.join(description_elements), source_url)
    description_text = clean_text(description_html)
    description_markdown = html_to_markdown(description_html)
    clean_desc = clean_text('\n'.join(description_elements))
    
    details.update({
        'description_html': description_html,
        'description_text': description_text,
        'description_markdown': description_markdown,
        'clean_desc': clean_desc
    })
    
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
                
                field_desc_html = clean_html(str(cols[desc_index]), source_url) if desc_index != -1 else None
                field_desc_text = clean_text(field_desc_html) if field_desc_html else None
                field_desc_markdown = html_to_markdown(field_desc_html) if field_desc_html else None
                field_clean_desc = clean_text(str(cols[desc_index])) if desc_index != -1 else None
                
                field = {
                    'name': cols[param_index].text.strip(),
                    'type': clean_html(str(cols[type_index]), source_url) if type_index != -1 else None,
                    'description_html': field_desc_html,
                    'description_text': field_desc_text,
                    'description_markdown': field_desc_markdown,
                    'clean_desc': field_clean_desc,
                    'required': cols[required_index].text.strip().lower() == 'yes' if required_index != -1 else False
                }
                details['fields'].append(field)
    
    return details

def clean_html(content, source_url):
    if not content:
        return ""
    soup = BeautifulSoup(content, 'html.parser')
    
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

def html_to_markdown(content):
    if not content:
        return ""
    
    content = re.sub(r'<br\s*/?>', '\n', content)
    content = re.sub(r'<strong>(.*?)</strong>', r'**\1**', content)
    content = re.sub(r'<b>(.*?)</b>', r'**\1**', content)
    content = re.sub(r'<em>(.*?)</em>', r'*\1*', content)
    content = re.sub(r'<i>(.*?)</i>', r'*\1*', content)
    content = re.sub(r'<code>(.*?)</code>', r'`\1`', content)
    content = re.sub(r'<pre>(.*?)</pre>', r'```\n\1\n```', content, flags=re.DOTALL)
    
    def replace_link(match):
        text = match.group(1)
        href = match.group(2)
        return f'[{text}]({href})'
    
    content = re.sub(r'<a\s+href="([^"]*)"[^>]*>(.*?)</a>', replace_link, content)
    
    content = CLEAN_PATTERN.sub('', content)
    
    return content.strip()

def generate_search_keywords(entity):
    keywords = set()
    name = entity['name'].lower()
    
    name_parts = NAME_PATTERN.findall(entity['name'])
    keywords.update(part.lower() for part in name_parts)
    keywords.add(name)
    
    if entity.get('clean_desc'):
        desc_keywords = set(WORD_PATTERN.findall(entity['clean_desc'].lower()))
        keywords.update(desc_keywords)
    
    for field in entity.get('fields', []):
        keywords.add(field['name'].lower())
        if field.get('clean_desc'):
            field_keywords = set(WORD_PATTERN.findall(field['clean_desc'].lower()))
            keywords.update(field_keywords)
    
    normalized_keywords = set()
    for kw in keywords:
        normalized = unidecode(kw)
        if 3 <= len(normalized) <= 50:
            normalized_keywords.add(normalized)
    
    return list(normalized_keywords)[:100]

def generate_cache_key(*args, **kwargs) -> str:
    key_str = f"{args}_{kwargs}"
    return hashlib.md5(key_str.encode()).hexdigest()

def search_in_memory(query: str, entity_type: Optional[str] = None, limit: int = 20) -> List[Dict]:
    query = query.lower().strip()
    if not query:
        return []
    
    results = []
    query_terms = query.split()
    
    for entity_id, entity in full_db_cache['entities'].items():
        if entity_type and entity['type'] != entity_type:
            continue
        
        score = 0
        entity_terms = full_db_cache['search_terms'].get(entity_id, [])
        
        if entity['name'].lower() == query:
            score += 1000
        
        if query in entity['name'].lower():
            score += 500
        
        for term, weight in entity_terms:
            for q_term in query_terms:
                if q_term in term:
                    score += weight * 10
                if term.startswith(q_term):
                    score += weight * 5
        
        if score > 0:
            fields = full_db_cache['entity_fields'].get(entity_id, [])
            notes = full_db_cache['entity_notes'].get(entity_id, [])
            
            results.append({
                **entity,
                'score': score,
                'fields': fields,
                'notes': notes,
                'match_terms': query_terms
            })
    
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:limit]

def get_formatted_content(entity: Dict, format_type: str = "normal") -> Dict[str, str]:
    if format_type == "html":
        return {
            "description": entity.get('description_html', ''),
            "fields": [
                {**field, 'description': field.get('description_html', '')} 
                for field in entity.get('fields', [])
            ],
            "notes": [
                {**note, 'content': note.get('html', '')} 
                for note in entity.get('notes', [])
            ]
        }
    elif format_type == "markdown":
        return {
            "description": entity.get('description_markdown', ''),
            "fields": [
                {**field, 'description': field.get('description_markdown', '')} 
                for field in entity.get('fields', [])
            ],
            "notes": [
                {**note, 'content': note.get('markdown', '')} 
                for note in entity.get('notes', [])
            ]
        }
    else:
        return {
            "description": entity.get('description_text', ''),
            "fields": [
                {**field, 'description': field.get('description_text', '')} 
                for field in entity.get('fields', [])
            ],
            "notes": [
                {**note, 'content': note.get('text', '')} 
                for note in entity.get('notes', [])
            ]
        }

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await initial_data_load()
    await load_full_db_cache()
    logger.info("🚀 Server started with full DB caching and initialization")
    yield
    await client.aclose()

app = FastAPI(
    title="Ultra Fast Telegram Bot API",
    description="Blazing fast Telegram Bot API search with proper initialization and notes support",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "status": "🚀 Ultra Fast Telegram Bot API",
        "entities": full_db_cache['entity_count'],
        "cache_stats": memory_cache.stats(),
        "last_updated": full_db_cache['last_updated'].isoformat()
    }

@app.get("/api/search")
async def search(
    q: str = Query(..., min_length=1, max_length=100),
    type: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    format: str = Query("normal")
):
    cache_key = generate_cache_key("search", q, type, limit, format)
    
    cached_result = memory_cache.get(cache_key)
    if cached_result:
        return JSONResponse(content=cached_result)
    
    try:
        start_time = datetime.now()
        results = search_in_memory(q, type, limit)
        search_time = (datetime.now() - start_time).total_seconds() * 1000
        
        formatted_results = []
        for result in results:
            formatted_content = get_formatted_content(result, format)
            
            formatted_results.append({
                "id": result["id"],
                "name": result["name"],
                "type": result["type"],
                "description": formatted_content["description"],
                "clean_desc": result.get('clean_desc', ''),
                "fields": formatted_content["fields"],
                "notes": formatted_content["notes"],
                "reference": f"{result['source_url']}#{result['name'].lower().replace(' ', '-')}",
                "score": result["score"]
            })
        
        response = {
            "query": q,
            "count": len(formatted_results),
            "results": formatted_results,
            "search_time_ms": search_time,
            "format": format
        }
        
        memory_cache.set(cache_key, response)
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Search failed")

@app.get("/api/types")
async def list_types():
    cache_key = "types_list"
    cached_result = memory_cache.get(cache_key)
    if cached_result:
        return JSONResponse(content=cached_result)
    
    types_count = {}
    for entity in full_db_cache['entities'].values():
        types_count[entity['type']] = types_count.get(entity['type'], 0) + 1
    
    types_list = [{"id": type_id, "name": type_id.title(), "count": count} 
                 for type_id, count in types_count.items()]
    
    memory_cache.set(cache_key, types_list)
    return JSONResponse(content=types_list)

@app.get("/api/entity/{entity_name}")
async def get_entity(entity_name: str, format: str = Query("normal")):
    cache_key = generate_cache_key("entity", entity_name, format)
    cached_result = memory_cache.get(cache_key)
    if cached_result:
        return JSONResponse(content=cached_result)
    
    entity = None
    for e in full_db_cache['entities'].values():
        if e['name'] == entity_name:
            entity_id = e['id']
            entity = e
            break
    
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    fields = full_db_cache['entity_fields'].get(entity_id, [])
    notes = full_db_cache['entity_notes'].get(entity_id, [])
    
    formatted_content = get_formatted_content(
        {**entity, 'fields': fields, 'notes': notes}, 
        format
    )
    
    response = {
        "id": entity_id,
        "name": entity["name"],
        "type": entity["type"],
        "description": formatted_content["description"],
        "clean_desc": entity.get('clean_desc', ''),
        "fields": formatted_content["fields"],
        "notes": formatted_content["notes"],
        "reference": f"{entity['source_url']}#{entity['name'].lower().replace(' ', '-')}"
    }
    
    memory_cache.set(cache_key, response)
    return JSONResponse(content=response)

@app.get("/api/list")
async def list_entities(type: str = Query(..., min_length=1)):
    cache_key = generate_cache_key("list", type)
    cached_result = memory_cache.get(cache_key)
    if cached_result:
        return JSONResponse(content=cached_result)
    
    entities = []
    for entity in full_db_cache['entities'].values():
        if entity['type'] == type:
            entities.append(entity['name'])
    
    entities.sort()
    memory_cache.set(cache_key, entities)
    return JSONResponse(content=entities)

@app.post("/api/update")
async def update_docs():
    try:
        success = await update_documentation_data()
        if success:
            await load_full_db_cache()
            memory_cache.clear()
            return {"status": "success", "entities": full_db_cache['entity_count']}
        else:
            raise HTTPException(status_code=500, detail="Update failed")
    except Exception as e:
        logger.error(f"Update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    return {
        "memory_cache": memory_cache.stats(),
        "database": {
            "entities": full_db_cache['entity_count'],
            "search_terms": full_db_cache['term_count'],
            "last_updated": full_db_cache['last_updated'].isoformat()
        }
    }

@app.post("/api/cache/clear")
async def clear_cache():
    memory_cache.clear()
    return {"status": "success", "message": "Cache cleared"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "entities": full_db_cache['entity_count'],
        "cache_hit_rate": memory_cache.stats()["hit_rate"],
        "timestamp": datetime.now(pytz.utc).isoformat()
    }

@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds() * 1000
    response.headers["X-Process-Time-MS"] = f"{process_time:.2f}"
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        workers=1,
        loop="uvloop",
        http="httptools",
        access_log=False
    )