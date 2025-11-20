from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
import sqlite3
import re
import httpx
from datetime import datetime, timedelta
import pytz
import threading
import time
import atexit
import asyncio
import logging
from difflib import get_close_matches
from unidecode import unidecode
import os
import json
from functools import lru_cache
from typing import Dict, List, Any, Optional
import hashlib
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
BOT_API_URL = 'https://core.telegram.org/bots/api'
WEBAPPS_URL = 'https://core.telegram.org/bots/webapps'
FEATURES_URL = 'https://core.telegram.org/bots/features'
FAQ_URL = 'https://core.telegram.org/bots/faq'
DB_FILE = 'bot_api.db'
CACHE_DIR = 'cache'
CACHE_DURATION = 3600  # 1 hour

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

def adapt_datetime_iso(val):
    return val.isoformat()

def convert_datetime_iso(val):
    return datetime.fromisoformat(val.decode())

sqlite3.register_adapter(datetime, adapt_datetime_iso)
sqlite3.register_converter("timestamp", convert_datetime_iso)

class CacheManager:
    @staticmethod
    def get_cache_key(*args, **kwargs) -> str:
        """Generate cache key from function arguments"""
        key_str = f"{args}_{kwargs}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    @staticmethod
    def get_cache_file(func_name: str, key: str) -> str:
        return os.path.join(CACHE_DIR, f"{func_name}_{key}.pkl")
    
    @staticmethod
    def is_cache_valid(cache_file: str, max_age: int = CACHE_DURATION) -> bool:
        if not os.path.exists(cache_file):
            return False
        file_time = os.path.getmtime(cache_file)
        return (time.time() - file_time) < max_age
    
    @staticmethod
    def get_cached_result(func_name: str, key: str):
        cache_file = CacheManager.get_cache_file(func_name, key)
        if CacheManager.is_cache_valid(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Cache read error for {func_name}: {e}")
        return None
    
    @staticmethod
    def set_cached_result(func_name: str, key: str, data):
        cache_file = CacheManager.get_cache_file(func_name, key)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Cache write error for {func_name}: {e}")

def cache_result(func):
    """Decorator to cache function results"""
    def wrapper(*args, **kwargs):
        # Skip caching for certain functions or conditions
        if kwargs.get('nocache') or getattr(func, '__nocache__', False):
            return func(*args, **kwargs)
            
        key = CacheManager.get_cache_key(*args, **kwargs)
        func_name = func.__name__
        
        # Try to get from cache
        cached_result = CacheManager.get_cached_result(func_name, key)
        if cached_result is not None:
            return cached_result
        
        # Execute function and cache result
        result = func(*args, **kwargs)
        if result is not None:
            CacheManager.set_cached_result(func_name, key, result)
        
        return result
    return wrapper

def init_db():
    with sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
        cursor = conn.cursor()
        # Enable WAL mode for better concurrent read performance
        cursor.execute('PRAGMA journal_mode=WAL')
        cursor.execute('PRAGMA synchronous=NORMAL')
        cursor.execute('PRAGMA cache_size=-64000')  # 64MB cache
        cursor.execute('PRAGMA temp_store=memory')
        
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
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entities_name_type ON entities(name, type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_fields_entity_id ON fields(entity_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_search_keyword ON search_index(keyword)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_search_entity_keyword ON search_index(entity_id, keyword)')
        
        cursor.execute('PRAGMA foreign_keys = ON')
        conn.commit()

# Global HTTP client with connection pooling
client = httpx.AsyncClient(
    timeout=30.0,
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
    http2=True
)

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
            # Only build content string when needed
            if len(current_section['elements']) < 50:  # Limit initial content building
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
    """Optimized entity details parsing"""
    if not content:
        return {'description': '', 'clean_desc': '', 'fields': []}
    
    soup = BeautifulSoup(content, 'html.parser')
    details = {'description': '', 'clean_desc': '', 'fields': []}
    
    # Find first table to stop description collection
    first_table = soup.find('table')
    description_elements = []
    
    for elem in soup.children:
        if elem == first_table:
            break
        description_elements.append(str(elem))
    
    details['description'] = clean_html('\n'.join(description_elements), source_url)
    details['clean_desc'] = clean_text('\n'.join(description_elements))
    
    # Parse tables more efficiently
    for table in soup.find_all('table'):
        headers = [th.text.strip().lower() for th in table.find_all('th')]
        if 'parameter' in headers or 'field' in headers:
            param_index = headers.index('parameter') if 'parameter' in headers else headers.index('field')
            type_index = headers.index('type') if 'type' in headers else -1
            required_index = headers.index('required') if 'required' in headers else -1
            desc_index = headers.index('description') if 'description' in headers else -1
            
            for row in table.find_all('tr')[1:]:  # Skip header row
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
    """Optimized keyword generation"""
    keywords = set()
    name = entity['name'].lower()
    
    # Add name variations
    name_parts = re.findall('[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', entity['name'])
    keywords.update(part.lower() for part in name_parts)
    keywords.add(name)
    
    # Add description keywords (limit to significant words)
    if entity.get('clean_desc'):
        desc_keywords = set(re.findall(r'\b\w{4,}\b', entity['clean_desc'].lower()))
        keywords.update(desc_keywords)
    
    # Add field names and types
    for field in entity.get('fields', []):
        keywords.add(field['name'].lower())
        if field.get('type'):
            # Extract text from HTML type field
            soup = BeautifulSoup(field['type'], 'html.parser')
            type_text = soup.get_text().lower()
            type_keywords = set(re.findall(r'\b\w{4,}\b', type_text))
            keywords.update(type_keywords)
    
    # Normalize and filter keywords
    normalized_keywords = set()
    for kw in keywords:
        normalized = unidecode(kw)
        if len(normalized) >= 3 and len(normalized) <= 50:  # Reasonable length limits
            normalized_keywords.add(normalized)
    
    return list(normalized_keywords)[:100]  # Limit keywords per entity

def html_to_markdown(html, source_url):
    if not html:
        return ""
    soup = BeautifulSoup(html, 'html.parser')
    
    # Process in order of specificity
    for pre in soup.find_all('pre'):
        code = pre.find('code')
        if code:
            pre.replace_with(f"```\n{code.get_text().strip()}\n```\n\n")
        else:
            pre.replace_with(f"```\n{pre.get_text().strip()}\n```\n\n")
    
    for code in soup.find_all('code'):
        if not code.find_parent('pre'):
            code.replace_with(f"`{code.get_text().strip()}`")
    
    for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        level = int(h.name[1])
        h.replace_with(f"{'#' * level} {h.get_text().strip()}\n\n")
    
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
    # Optimized text cleaning
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n[ \t]+', '\n', text)
    
    return text.strip()

def html_to_telegram_mdv1(html, source_url):
    if not html:
        return ""
    soup = BeautifulSoup(html, 'html.parser')
    
    # Process code blocks first
    for pre in soup.find_all('pre'):
        code = pre.get_text().strip()
        pre.replace_with(f"```\n{code}\n```\n\n")
    
    for code in soup.find_all('code'):
        if not code.find_parent('pre'):
            code.replace_with(f"`{code.get_text(strip=True)}`")
    
    for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        h.replace_with(f"\n*{h.get_text(strip=True)}*\n\n")
    
    for b in soup.find_all(['b', 'strong']):
        b.replace_with(f"*{b.get_text(strip=True)}*")
    
    for i in soup.find_all(['i', 'em']):
        i.replace_with(f"_{i.get_text(strip=True)}_")
    
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
    """Optimized documentation update with batch operations"""
    now = datetime.now(pytz.utc)
    
    with sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
        cursor = conn.cursor()
        
        # Use transaction for better performance
        cursor.execute("BEGIN TRANSACTION")
        
        # Clear old data
        cursor.execute("DELETE FROM entities WHERE source_url IN (?, ?, ?, ?)", 
                      (BOT_API_URL, WEBAPPS_URL, FEATURES_URL, FAQ_URL))
        
        # Fetch all sources concurrently
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
                entity_type = determine_entity_type(section['title'], source_url)
                if entity_type == 'other':
                    continue
                
                # Build full content only when needed
                if not section.get('content') and section.get('elements'):
                    section['content'] = ''.join(str(elem) for elem in section['elements'])
                
                details = parse_entity_details(section['content'], entity_type, source_url)
                
                # Insert entity
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
                entity_count += 1
                
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
        
        cursor.execute("COMMIT")
        logger.info(f"Updated {entity_count} entities")
    
    # Clear search cache after update
    clear_search_cache()
    return True

def clear_search_cache():
    """Clear all search-related cache files"""
    try:
        for filename in os.listdir(CACHE_DIR):
            if filename.startswith(('search_', 'list_entities_')):
                os.remove(os.path.join(CACHE_DIR, filename))
        logger.info("Search cache cleared")
    except Exception as e:
        logger.warning(f"Error clearing cache: {e}")

def initial_load():
    """Optimized initial load with progress tracking"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        db_exists = os.path.exists(DB_FILE)
        needs_load = False
        
        if db_exists:
            with sqlite3.connect(DB_FILE) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM entities')
                entity_count = cursor.fetchone()[0]
                needs_load = entity_count == 0
        else:
            needs_load = True
            
        if needs_load:
            logger.info("Performing initial database load")
            success = loop.run_until_complete(update_docs())
            if success:
                logger.info("Initial database load completed successfully")
            else:
                logger.error("Initial database load failed")
        else:
            logger.info(f"Database already exists with {entity_count} entities")
    finally:
        loop.close()

# Initialize database and load data
init_db()
initial_load()

def cleanup():
    asyncio.run(client.aclose())

atexit.register(cleanup)

@app.route('/api/search', methods=['GET'])
@cache_result
def search():
    """Optimized search with caching"""
    query = request.args.get('q', '').strip().lower()
    entity_type = request.args.get('type')
    limit = int(request.args.get('limit', 20))
    format_type = request.args.get('format', 'normal')
    
    if format_type not in ['normal', 'markdown', 'html']:
        return jsonify({"error": "Invalid format parameter. Use 'normal', 'markdown' or 'html'"}), 400
    
    if not query:
        return jsonify({"error": "Missing search query"}), 400
    
    try:
        with sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Exact matches (most relevant)
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
            
            # Keyword matches with scoring
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
            
            # Similar matches using fuzzy matching
            similar_results = []
            if len(exact_matches) + len(keyword_matches) < limit:
                cursor.execute('SELECT name, type FROM entities')
                all_entities = [(row[0].lower(), row[1]) for row in cursor.fetchall()]
                
                if entity_type:
                    all_names = [name for name, typ in all_entities if typ == entity_type]
                else:
                    all_names = [name for name, typ in all_entities]
                
                similar_names = get_close_matches(query, all_names, n=5, cutoff=0.3)
                
                if similar_names:
                    placeholders = ','.join(['?'] * len(similar_names))
                    similar_query = f'''
                    SELECT e.id, e.name, e.type, e.description, e.clean_desc, 
                           COALESCE(e.content, '') as content, e.source_url
                    FROM entities e
                    WHERE LOWER(e.name) IN ({placeholders})
                    '''
                    if entity_type:
                        similar_query += ' AND e.type = ?'
                        similar_params = similar_names + [entity_type]
                    else:
                        similar_params = similar_names
                    
                    cursor.execute(similar_query, similar_params)
                    similar_results = [dict(row) for row in cursor.fetchall()]
            
            # Combine and deduplicate results
            combined_results = []
            seen_ids = set()
            
            for match in exact_matches + keyword_matches + similar_results:
                if match['id'] not in seen_ids and len(combined_results) < limit:
                    seen_ids.add(match['id'])
                    combined_results.append(match)
            
            # Format results
            results = []
            for result in combined_results:
                # Get fields for this entity
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
            
            return response
            
    except Exception as e:
        logger.error(f"Search failed: {str(e)}", exc_info=True)
        return {"error": "Internal server error"}

@app.route('/api/types', methods=['GET'])
@cache_result
def list_types():
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

@app.route('/api/list', methods=['GET'])
@cache_result
def list_entities():
    entity_type = request.args.get('type', '').strip().lower()
    
    if not entity_type:
        return {"error": "Missing type parameter"}
    
    valid_types = ['method', 'object', 'webapp', 'feature', 'faq', 'other']
    if entity_type not in valid_types:
        return {"error": f"Invalid type. Valid types are: {', '.join(valid_types)}"}
    
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
        return {"error": "Internal server error"}

@app.route('/api/update', methods=['POST'])
def update_docs_endpoint():
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(update_docs())
        loop.close()
        if success:
            return jsonify({"status": "success"})
        else:
            return jsonify({"status": "failed"}), 500
    except Exception as e:
        logger.error(f"Failed to update docs: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/cache/clear', methods=['POST'])
def clear_cache_endpoint():
    """Endpoint to manually clear cache"""
    try:
        clear_search_cache()
        return jsonify({"status": "success", "message": "Cache cleared"})
    except Exception as e:
        logger.error(f"Failed to clear cache: {str(e)}")
        return jsonify({"error": "Failed to clear cache"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for deployment"""
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM entities')
            entity_count = cursor.fetchone()[0]
        
        return jsonify({
            "status": "healthy",
            "entities": entity_count,
            "timestamp": datetime.now(pytz.utc).isoformat()
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)