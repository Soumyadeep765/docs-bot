import os
import sqlite3
import logging
import asyncio
import aiohttp
from datetime import datetime
from aiohttp import web
import pytz
from bs4 import BeautifulSoup
import re
from unidecode import unidecode
from difflib import get_close_matches
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_FILE = 'bot_api.db'
BOT_TOKEN = os.getenv('BOT_TOKEN', '7828383323:AAFldn_KaEDMurNgC0pM-z4lAxd39UNpzNU')
BOT_API_URL = 'https://core.telegram.org/bots/api'
WEBAPPS_URL = 'https://core.telegram.org/bots/webapps'
FEATURES_URL = 'https://core.telegram.org/bots/features'
FAQ_URL = 'https://core.telegram.org/bots/faq'

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
        cursor.execute('PRAGMA foreign_keys = ON')
        conn.commit()

async def fetch_url(session, url):
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            soup = BeautifulSoup(await response.text(), 'html.parser')
            content_div = soup.find('div', class_='dev_page_content') or soup.find(id='dev_page_content')
            return content_div
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {str(e)}")
        return None

async def fetch_bot_api(session):
    content_div = await fetch_url(session, BOT_API_URL)
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
                'anchor': element.get('id', '').strip()
            }
        elif current_section:
            current_section['elements'].append(element)
            current_section['content'] += str(element)
    if current_section:
        sections.append(current_section)
    return sections

async def fetch_webapps(session):
    content_div = await fetch_url(session, WEBAPPS_URL)
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
                'anchor': element.get('id', '').strip()
            }
        elif current_section:
            current_section['elements'].append(element)
            current_section['content'] += str(element)
    if current_section:
        sections.append(current_section)
    return sections

async def fetch_features(session):
    content_div = await fetch_url(session, FEATURES_URL)
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
                'anchor': element.get('id', '').strip()
            }
        elif current_section:
            current_section['elements'].append(element)
            current_section['content'] += str(element)
    if current_section:
        sections.append(current_section)
    return sections

async def fetch_faq(session):
    content_div = await fetch_url(session, FAQ_URL)
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
                'anchor': element.get('id', '').strip()
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

def format_content(content, source_url):
    if not content:
        return ""
    return html_to_telegram_mdv1(content, source_url)

def generate_reference_url(name, source_url):
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
    async with aiohttp.ClientSession() as session:
        with sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM entities WHERE source_url IN (?, ?, ?, ?)", 
                          (BOT_API_URL, WEBAPPS_URL, FEATURES_URL, FAQ_URL))
            cursor.execute("DELETE FROM fields WHERE entity_id NOT IN (SELECT id FROM entities)")
            cursor.execute("DELETE FROM search_index WHERE entity_id NOT IN (SELECT id FROM entities)")
            
            sources = [
                (BOT_API_URL, await fetch_bot_api(session)),
                (WEBAPPS_URL, await fetch_webapps(session)),
                (FEATURES_URL, await fetch_features(session)),
                (FAQ_URL, await fetch_faq(session))
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
                    for field in details.get('fields', []):
                        cursor.execute(
                            '''INSERT INTO fields (entity_id, name, type, description, clean_desc, required)
                            VALUES (?, ?, ?, ?, ?, ?)''',
                            (
                                entity_id, 
                                field['name'], 
                                field.get('type'), 
                                field.get('description'), 
                                field.get('clean_desc'), 
                                field.get('required', False)
                            ))
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
                        cursor.execute(
                            "INSERT INTO search_index (entity_id, keyword, weight) VALUES (?, ?, ?)",
                            (entity_id, keyword, weight)
                        )
            conn.commit()
    return True

def initial_load():
    if not os.path.exists(DB_FILE):
        logger.info("Initial database load required")
        asyncio.run(update_docs())
    else:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT 1 FROM entities LIMIT 1')
            has_data = cursor.fetchone() is not None
        if not has_data:
            logger.info("Initial database load required")
            asyncio.run(update_docs())
        else:
            logger.info("Database already exists with data")

init_db()
initial_load()

async def search_entities(query, entity_type=None, limit=20):
    query = query.strip().lower()
    if not query:
        return []
    
    try:
        with sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            exact_query = '''
            SELECT e.id, e.name, e.type, e.description, e.clean_desc, COALESCE(e.content, '') as content, e.source_url
            FROM entities e
            WHERE LOWER(e.name) = ?
            '''
            exact_params = [query]
            if entity_type:
                exact_query += ' AND e.type = ?'
                exact_params.append(entity_type)
            
            cursor.execute(exact_query, exact_params)
            exact_matches = [dict(row) for row in cursor.fetchall()]
            
            keyword_query = '''
            SELECT e.id, e.name, e.type, e.description, e.clean_desc, COALESCE(e.content, '') as content, SUM(si.weight) as score, e.source_url
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
            
            cursor.execute('SELECT name, type FROM entities')
            all_entities = [(row[0].lower(), row[1]) for row in cursor.fetchall()]
            
            if entity_type:
                all_names = [name for name, typ in all_entities if typ == entity_type]
            else:
                all_names = [name for name, typ in all_entities]
            
            similar_names = get_close_matches(query, all_names, n=5, cutoff=0.3)
            
            similar_results = []
            if similar_names:
                placeholders = ','.join(['?'] * len(similar_names))
                similar_query = f'''
                SELECT e.id, e.name, e.type, e.description, e.clean_desc, COALESCE(e.content, '') as content, e.source_url
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
            
            combined_results = []
            seen_ids = set()
            
            for match in exact_matches:
                if match['id'] not in seen_ids:
                    seen_ids.add(match['id'])
                    combined_results.append(match)
            
            for match in keyword_matches:
                if match['id'] not in seen_ids:
                    seen_ids.add(match['id'])
                    combined_results.append(match)
            
            for match in similar_results:
                if match['id'] not in seen_ids:
                    seen_ids.add(match['id'])
                    combined_results.append(match)
            
            results = []
            for result in combined_results[:limit]:
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
                        'type': format_content(field['type'], result['source_url']),
                        'description': format_content(field['description'], result['source_url']),
                        'clean_desc': field['clean_desc'],
                        'required': field['required']
                    })
                
                results.append({
                    'id': result['id'],
                    'name': result['name'],
                    'type': result['type'],
                    'description': format_content(result.get('description'), result['source_url']),
                    'clean_desc': result.get('clean_desc'),
                    'content': format_content(result.get('content'), result['source_url']),
                    'fields': formatted_fields,
                    'reference': generate_reference_url(result['name'], result['source_url']),
                    'match_type': 'exact' if result in exact_matches else 
                                'keyword' if result in keyword_matches else 
                                'similar'
                })
            
            return results
            
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        return []

async def handle_search(request):
    query = request.query.get('q', '').strip()
    entity_type = request.query.get('type')
    limit = int(request.query.get('limit', 20))
    
    if not query:
        return web.json_response({"error": "Missing search query"}, status=400)
    
    results = await search_entities(query, entity_type, limit)
    
    response = {
        'query': query,
        'count': len(results),
        'results': results
    }
    
    return web.json_response(response)

async def handle_webhook(request):
    if request.method == 'POST':
        try:
            data = await request.json()
            update_id = data.get('update_id')
            
            if 'inline_query' in data:
                inline_query = data['inline_query']
                query_id = inline_query['id']
                query_text = inline_query.get('query', '').strip()
                
                if not query_text:
                    results = [{
                        "type": "article",
                        "id": "start",
                        "title": "🔍 Telegram Bot API Search",
                        "description": "Enter a method or object name to search documentation",
                        "input_message_content": {
                            "message_text": "🔍 *Telegram Bot API Search*\n\nEnter a method or object name to search the official documentation.\n\nExamples: `sendMessage`, `ReplyKeyboardMarkup`, `WebApp`",
                            "parse_mode": "Markdown"
                        }
                    }]
                else:
                    search_results = await search_entities(query_text, limit=50)
                    
                    if not search_results:
                        results = [{
                            "type": "article",
                            "id": "no_results",
                            "title": "😔 No Results Found",
                            "description": "Try searching with a different keyword",
                            "input_message_content": {
                                "message_text": f"❌ *No matching documentation found for:* `{query_text}`\n\n🔁 Try again with a more specific keyword.",
                                "parse_mode": "Markdown"
                            },
                            "reply_markup": {
                                "inline_keyboard": [[
                                    {"text": "🔍 Search Again", "switch_inline_query_current_chat": ""}
                                ]]
                            }
                        }]
                    else:
                        results = []
                        for item in search_results[:50]:
                            name = item['name']
                            description = item.get('clean_desc', '')[:100] + '...' if len(item.get('clean_desc', '')) > 100 else item.get('clean_desc', '')
                            
                            message_text = f"*{name}*\n\n{item['description']}"
                            if len(message_text) > 4096:
                                message_text = message_text[:4000] + "...\n\n📖 *Read more in the official documentation*"
                            
                            results.append({
                                "type": "article",
                                "id": str(item['id']),
                                "title": name,
                                "description": description,
                                "input_message_content": {
                                    "message_text": message_text,
                                    "parse_mode": "Markdown",
                                    "disable_web_page_preview": True
                                },
                                "reply_markup": {
                                    "inline_keyboard": [[
                                        {"text": "📖 Open Docs", "url": item['reference']},
                                        {"text": "🔍 Search Again", "switch_inline_query_current_chat": ""}
                                    ]]
                                }
                            })
                
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        f'https://api.telegram.org/bot{BOT_TOKEN}/answerInlineQuery',
                        json={
                            'inline_query_id': query_id,
                            'results': results,
                            'cache_time': 1800,
                            'is_personal': True
                        }
                    )
            
            return web.Response(text='OK')
            
        except Exception as e:
            logger.error(f"Webhook error: {str(e)}")
            return web.Response(text='ERROR', status=500)
    
    return web.Response(text='Method not allowed', status=405)

async def handle_update_docs(request):
    if request.method == 'POST':
        try:
            success = await update_docs()
            if success:
                return web.json_response({"status": "success"})
            else:
                return web.json_response({"status": "failed"}, status=500)
        except Exception as e:
            logger.error(f"Failed to update docs: {str(e)}")
            return web.json_response({"error": "Internal server error"}, status=500)
    
    return web.Response(text='Method not allowed', status=405)

app = web.Application()
app.router.add_get('/api/search', handle_search)
app.router.add_post('/webhook', handle_webhook)
app.router.add_post('/api/update', handle_update_docs)

if __name__ == '__main__':
    web.run_app(app, host='0.0.0.0', port=5000)