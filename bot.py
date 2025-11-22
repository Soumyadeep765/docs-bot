import asyncio
import logging
import re
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any
from urllib.parse import quote

import aiosqlite
import httpx
from bs4 import BeautifulSoup
from telegram import (
    InlineQueryResultArticle,
    InputTextMessageContent,
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    WebAppInfo
)
from telegram.ext import (
    Application,
    CommandHandler,
    InlineQueryHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters
)
from unidecode import unidecode

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration
BOT_API_URL = 'https://core.telegram.org/bots/api'
WEBAPPS_URL = 'https://core.telegram.org/bots/webapps'
FEATURES_URL = 'https://core.telegram.org/bots/features'
FAQ_URL = 'https://core.telegram.org/bots/faq'
DB_FILE = 'telegram_bot_api.db'

# Pre-compiled regex patterns
CLEAN_PATTERN = re.compile(r'<.*?>')
WORD_PATTERN = re.compile(r'\b\w{3,}\b')
NAME_PATTERN = re.compile(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)')
NOTE_PATTERN = re.compile(r'<strong>(\d+)\.</strong>\s*(.*?)(?=<strong>\d+\.</strong>|</blockquote>|$)', re.DOTALL)
BLOCKQUOTE_PATTERN = re.compile(r'<blockquote>(.*?)</blockquote>', re.DOTALL)
SYMBOL_PATTERN = re.compile(r'\$(\w+)')
DOUBLE_SYMBOL_PATTERN = re.compile(r'\$\$(\w+)')

class TelegramBotAPISearch:
    def __init__(self):
        self.db_file = DB_FILE
        self.full_cache = {}
        self.http_client = None
        
    async def init(self):
        """Initialize the search system"""
        await self.init_db()
        await self.load_data()
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
    async def init_db(self):
        """Initialize database"""
        async with aiosqlite.connect(self.db_file) as db:
            await db.execute('PRAGMA journal_mode=WAL')
            await db.execute('PRAGMA synchronous=NORMAL')
            await db.execute('PRAGMA cache_size=-100000')
            
            await db.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                description_html TEXT,
                description_text TEXT,
                clean_desc TEXT,
                source_url TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(name, type)
            )
            ''')
            
            await db.execute('''
            CREATE TABLE IF NOT EXISTS fields (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id INTEGER,
                name TEXT NOT NULL,
                type TEXT,
                description_text TEXT,
                clean_desc TEXT,
                required INTEGER DEFAULT 0,
                FOREIGN KEY (entity_id) REFERENCES entities(id),
                UNIQUE(entity_id, name)
            )
            ''')
            
            await db.execute('''
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id INTEGER,
                note_number INTEGER,
                content_text TEXT,
                clean_desc TEXT,
                FOREIGN KEY (entity_id) REFERENCES entities(id),
                UNIQUE(entity_id, note_number)
            )
            ''')
            
            await db.execute('''
            CREATE TABLE IF NOT EXISTS search_index (
                entity_id INTEGER,
                keyword TEXT,
                weight INTEGER DEFAULT 1,
                FOREIGN KEY (entity_id) REFERENCES entities(id),
                PRIMARY KEY (entity_id, keyword)
            )
            ''')
            
            await db.execute('CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_search_keyword ON search_index(keyword)')
            
            await db.commit()
    
    async def load_data(self):
        """Load data into memory cache"""
        async with aiosqlite.connect(self.db_file) as db:
            db.row_factory = aiosqlite.Row
            
            # Load entities
            entities = {}
            async with db.execute('SELECT * FROM entities') as cursor:
                async for row in cursor:
                    entities[row['id']] = dict(row)
            
            # Load search index
            search_terms = {}
            async with db.execute('SELECT * FROM search_index') as cursor:
                async for row in cursor:
                    entity_id = row['entity_id']
                    if entity_id not in search_terms:
                        search_terms[entity_id] = []
                    search_terms[entity_id].append((row['keyword'], row['weight']))
            
            # Load fields
            entity_fields = {}
            async with db.execute('SELECT * FROM fields') as cursor:
                async for row in cursor:
                    entity_id = row['entity_id']
                    if entity_id not in entity_fields:
                        entity_fields[entity_id] = []
                    entity_fields[entity_id].append(dict(row))
            
            # Load notes
            entity_notes = {}
            async with db.execute('SELECT * FROM notes') as cursor:
                async for row in cursor:
                    entity_id = row['entity_id']
                    if entity_id not in entity_notes:
                        entity_notes[entity_id] = []
                    entity_notes[entity_id].append(dict(row))
            
            self.full_cache = {
                'entities': entities,
                'search_terms': search_terms,
                'entity_fields': entity_fields,
                'entity_notes': entity_notes,
                'last_updated': datetime.now()
            }
    
    async def update_documentation(self):
        """Update documentation from sources"""
        try:
            async with aiosqlite.connect(self.db_file) as db:
                await db.execute('DELETE FROM entities')
                await db.execute('DELETE FROM fields')
                await db.execute('DELETE FROM notes')
                await db.execute('DELETE FROM search_index')
                await db.commit()
            
            sources = [
                (BOT_API_URL, await self.fetch_bot_api()),
                (WEBAPPS_URL, await self.fetch_webapps()),
                (FEATURES_URL, await self.fetch_features()),
                (FAQ_URL, await self.fetch_faq())
            ]
            
            for source_url, sections in sources:
                if sections:
                    for section in sections:
                        await self.save_entity(section, source_url)
            
            await self.load_data()
            return True
        except Exception as e:
            logger.error(f"Update failed: {e}")
            return False
    
    async def fetch_bot_api(self):
        """Fetch Bot API documentation"""
        try:
            async with self.http_client as client:
                response = await client.get(BOT_API_URL)
                soup = BeautifulSoup(response.text, 'html.parser')
                content_div = soup.find('div', class_='dev_page_content')
                return self.parse_sections(content_div, ['h3', 'h4']) if content_div else []
        except Exception as e:
            logger.error(f"Failed to fetch Bot API: {e}")
            return []
    
    async def fetch_webapps(self):
        """Fetch WebApps documentation"""
        try:
            async with self.http_client as client:
                response = await client.get(WEBAPPS_URL)
                soup = BeautifulSoup(response.text, 'html.parser')
                content_div = soup.find('div', class_='dev_page_content')
                return self.parse_sections(content_div, ['h2', 'h3', 'h4']) if content_div else []
        except Exception as e:
            logger.error(f"Failed to fetch WebApps: {e}")
            return []
    
    async def fetch_features(self):
        """Fetch Features documentation"""
        try:
            async with self.http_client as client:
                response = await client.get(FEATURES_URL)
                soup = BeautifulSoup(response.text, 'html.parser')
                content_div = soup.find('div', class_='dev_page_content')
                return self.parse_sections(content_div, ['h2', 'h3', 'h4']) if content_div else []
        except Exception as e:
            logger.error(f"Failed to fetch Features: {e}")
            return []
    
    async def fetch_faq(self):
        """Fetch FAQ documentation"""
        try:
            async with self.http_client as client:
                response = await client.get(FAQ_URL)
                soup = BeautifulSoup(response.text, 'html.parser')
                content_div = soup.find('div', class_='dev_page_content')
                return self.parse_sections(content_div, ['h3', 'h4']) if content_div else []
        except Exception as e:
            logger.error(f"Failed to fetch FAQ: {e}")
            return []
    
    def parse_sections(self, content_div, heading_tags):
        """Parse content into sections"""
        sections = []
        current_section = None
        
        for element in content_div.children:
            if element.name in heading_tags:
                if current_section:
                    sections.append(current_section)
                current_section = {
                    'title': element.get_text().strip(),
                    'content': '',
                    'elements': []
                }
            elif current_section:
                current_section['elements'].append(element)
                current_section['content'] += str(element)
        
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def determine_entity_type(self, title, source_url):
        """Determine entity type"""
        if not title:
            return 'other'
        
        title = title.strip().lower()
        if source_url == BOT_API_URL:
            if title and title[0].islower():
                return 'method'
            if title and title[0].isupper():
                return 'object'
            if 'webapp' in title:
                return 'webapp'
        elif source_url == WEBAPPS_URL:
            return 'webapp'
        elif source_url == FEATURES_URL:
            return 'feature'
        elif source_url == FAQ_URL:
            return 'faq'
        return 'other'
    
    async def save_entity(self, section, source_url):
        """Save entity to database"""
        entity_type = self.determine_entity_type(section['title'], source_url)
        if entity_type == 'other':
            return
        
        if not section.get('content') and section.get('elements'):
            section['content'] = ''.join(str(elem) for elem in section['elements'])
        
        cleaned_content, notes = self.extract_notes(section['content'])
        details = self.parse_entity_details(cleaned_content, source_url)
        
        async with aiosqlite.connect(self.db_file) as db:
            # Insert entity
            cursor = await db.execute(
                '''INSERT INTO entities (name, type, description_html, description_text, clean_desc, source_url)
                VALUES (?, ?, ?, ?, ?, ?)''',
                (
                    section['title'],
                    entity_type,
                    details['description_html'],
                    details['description_text'],
                    details['clean_desc'],
                    source_url
                )
            )
            entity_id = cursor.lastrowid
            
            # Insert notes
            for note in notes:
                await db.execute(
                    'INSERT INTO notes (entity_id, note_number, content_text, clean_desc) VALUES (?, ?, ?, ?)',
                    (entity_id, note['number'], note['text'], note['clean_desc'])
                )
            
            # Insert fields
            for field in details.get('fields', []):
                await db.execute(
                    '''INSERT INTO fields (entity_id, name, type, description_text, clean_desc, required)
                    VALUES (?, ?, ?, ?, ?, ?)''',
                    (
                        entity_id,
                        field['name'],
                        field.get('type'),
                        field.get('description_text'),
                        field.get('clean_desc'),
                        field.get('required', False)
                    )
                )
            
            # Generate search keywords
            keywords = self.generate_search_keywords({
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
    
    def extract_notes(self, content):
        """Extract notes from content"""
        if not content:
            return content, []
        
        notes = []
        blockquote_matches = BLOCKQUOTE_PATTERN.findall(content)
        
        for blockquote in blockquote_matches:
            note_matches = NOTE_PATTERN.findall(blockquote)
            
            for note_num, note_content in note_matches:
                note_num = int(note_num.strip())
                note_content = note_content.strip()
                
                text_content = self.clean_text(f"{note_num}. {note_content}")
                clean_desc = self.clean_text(note_content)
                
                notes.append({
                    'number': note_num,
                    'text': text_content,
                    'clean_desc': clean_desc
                })
        
        cleaned_content = BLOCKQUOTE_PATTERN.sub('', content)
        return cleaned_content, notes
    
    def parse_entity_details(self, content, source_url):
        """Parse entity details"""
        if not content:
            return {
                'description_html': '',
                'description_text': '',
                'clean_desc': '',
                'fields': []
            }
        
        soup = BeautifulSoup(content, 'html.parser')
        details = {
            'description_html': '',
            'description_text': '',
            'clean_desc': '',
            'fields': []
        }
        
        description_elements = []
        for elem in soup.children:
            if elem.name == 'table':
                break
            description_elements.append(str(elem))
        
        description_html = self.clean_html('\n'.join(description_elements), source_url)
        description_text = self.clean_text(description_html)
        clean_desc = self.clean_text('\n'.join(description_elements))
        
        details.update({
            'description_html': description_html,
            'description_text': description_text,
            'clean_desc': clean_desc
        })
        
        # Parse fields from tables
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
                    
                    field_desc_text = self.clean_text(str(cols[desc_index])) if desc_index != -1 else None
                    field_clean_desc = self.clean_text(str(cols[desc_index])) if desc_index != -1 else None
                    
                    field = {
                        'name': cols[param_index].text.strip(),
                        'type': self.clean_html(str(cols[type_index]), source_url) if type_index != -1 else None,
                        'description_text': field_desc_text,
                        'clean_desc': field_clean_desc,
                        'required': cols[required_index].text.strip().lower() == 'yes' if required_index != -1 else False
                    }
                    details['fields'].append(field)
        
        return details
    
    def clean_html(self, content, source_url):
        """Clean HTML content"""
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
    
    def clean_text(self, content):
        """Extract clean text from HTML"""
        if not content:
            return ""
        soup = BeautifulSoup(content, 'html.parser')
        return soup.get_text().strip()
    
    def generate_search_keywords(self, entity):
        """Generate search keywords"""
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
    
    def search_entities(self, query: str, entity_type: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Search entities in memory"""
        query = query.lower().strip()
        if not query:
            return []
        
        # Handle special commands
        if query.startswith('!') and ' ' in query:
            parts = query.split(' ', 1)
            entity_type = parts[0][1:]  # Remove !
            query = parts[1]
        
        results = []
        query_terms = query.split()
        
        for entity_id, entity in self.full_cache['entities'].items():
            if entity_type and entity['type'] != entity_type:
                continue
            
            score = 0
            entity_terms = self.full_cache['search_terms'].get(entity_id, [])
            
            # Exact matches
            if entity['name'].lower() == query:
                score += 1000
            
            if query in entity['name'].lower():
                score += 500
            
            # Term matches
            for term, weight in entity_terms:
                for q_term in query_terms:
                    if q_term in term:
                        score += weight * 10
                    if term.startswith(q_term):
                        score += weight * 5
            
            if score > 0:
                fields = self.full_cache['entity_fields'].get(entity_id, [])
                notes = self.full_cache['entity_notes'].get(entity_id, [])
                
                results.append({
                    **entity,
                    'id': entity_id,
                    'score': score,
                    'fields': fields,
                    'notes': notes
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]
    
    def format_symbols(self, text: str) -> str:
        """Format API symbols in text"""
        def replace_symbol(match):
            symbol = match.group(1)
            # Search for the symbol
            results = self.search_entities(symbol, limit=1)
            if results:
                return f'<a href="{results[0]["source_url"]}">{symbol}</a>'
            return symbol
        
        # Handle $$ symbols (don't replace)
        text = DOUBLE_SYMBOL_PATTERN.sub(r'\1', text)
        # Handle $ symbols
        text = SYMBOL_PATTERN.sub(replace_symbol, text)
        
        return text
    
    def sanitize_html(self, text: str) -> str:
        """Sanitize HTML for Telegram"""
        if not text:
            return ''
        
        allowed_tags = ['b', 'strong', 'i', 'em', 'u', 's', 'strike', 'del', 'code', 'pre', 'a', 'tg-spoiler']
        
        def sanitize_tag(match):
            slash, tag, attrs = match.groups()
            if tag.lower() in allowed_tags:
                # Only allow href and class attributes for links
                if tag.lower() == 'a':
                    href_match = re.search(r'href=(["\'])(.*?)\1', attrs)
                    if href_match:
                        href = href_match.group(2)
                        return f'<{slash}{tag} href="{href}">'
                return f'<{slash}{tag}{attrs}>'
            return ''
        
        text = re.sub(r'<!--[\s\S]*?-->', '', text)
        text = re.sub(r'<(\/?)([a-zA-Z0-9_-]+)([^>]*)>', sanitize_tag, text)
        
        return text
    
    def truncate_safe_html(self, html: str, max_len: int = 4000) -> str:
        """Safely truncate HTML without breaking tags"""
        if len(html) <= max_len:
            return html
        
        output = ''
        length = 0
        tag_stack = []
        parts = re.split(r'(<[^>]+>)', html)
        
        for part in parts:
            if not part:
                continue
            
            if part.startswith('<'):
                output += part
                tag_match = re.match(r'^<\s*\/?([a-zA-Z0-9]+).*?>$', part)
                if tag_match:
                    tag = tag_match.group(1)
                    if part.startswith('</'):
                        if tag_stack and tag_stack[-1] == tag:
                            tag_stack.pop()
                    else:
                        tag_stack.append(tag)
            else:
                remaining = max_len - length
                if remaining <= 0:
                    break
                sliced = part[:remaining]
                output += sliced
                length += len(sliced)
                if len(sliced) < len(part):
                    break
        
        # Close any open tags
        for tag in reversed(tag_stack):
            output += f'</{tag}>'
        
        return output + '...\n\n<i>More⬇️👇</i>'

# Initialize the search system
search_system = TelegramBotAPISearch()

# Bot handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send welcome message"""
    welcome_text = """
👋 <b>Welcome to TgDocsXBot!</b>

Use me inline anywhere to search documentation fast.

🪄 <b>Just type:</b> @TgDocsXBot something

📚 <b>Example:</b> @TgDocsXBot getUpdates

<b>Special Features:</b>
• <code>!type</code> - Filter by category (method, object, webapp, etc.)
• <code>.property</code> - Search properties/parameters
• <code>$Symbol</code> - Format API symbols in text

<b>Examples:</b>
<code>@TgDocsXBot !method send</code> - Find send methods
<code>@TgDocsXBot .f message_id</code> - Find message_id property
    """
    
    keyboard = [
        [InlineKeyboardButton("🔍 Search Examples", switch_inline_query_current_chat="getUpdates")],
        [InlineKeyboardButton("📚 Categories", callback_data="categories"),
         InlineKeyboardButton("🆘 Help", callback_data="help")],
        [InlineKeyboardButton("🔄 Update Docs", callback_data="update_docs")]
    ]
    
    await update.message.reply_text(
        welcome_text,
        parse_mode='HTML',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show help"""
    help_text = """
<b>📖 TgDocsXBot Help</b>

<b>Basic Usage:</b>
Type <code>@TgDocsXBot query</code> in any chat

<b>Advanced Search:</b>
• <code>!method</code> - Bot API methods
• <code>!object</code> - Bot API objects  
• <code>!webapp</code> - WebApp features
• <code>!feature</code> - Bot features
• <code>!faq</code> - Frequently asked questions

<b>Symbol Formatting:</b>
Use <code>$Symbol</code> to automatically link API symbols:
<code>Use $Message and $Chat</code> becomes linked text

<b>Properties Search:</b>
Use <code>.property</code> to search specific properties

<b>Examples:</b>
<code>@TgDocsXBot !method send</code>
<code>@TgDocsXBot Message .message_id</code>
<code>@TgDocsXBot $Message $Chat</code>
    """
    
    keyboard = [
        [InlineKeyboardButton("🔍 Try Search", switch_inline_query_current_chat="")],
        [InlineKeyboardButton("📚 Categories", callback_data="categories"),
         InlineKeyboardButton("⬅️ Back", callback_data="back_to_start")]
    ]
    
    if update.callback_query:
        await update.callback_query.edit_message_text(
            help_text,
            parse_mode='HTML',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    else:
        await update.message.reply_text(
            help_text,
            parse_mode='HTML',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

async def show_categories(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show available categories"""
    categories = {}
    for entity in search_system.full_cache['entities'].values():
        categories[entity['type']] = categories.get(entity['type'], 0) + 1
    
    categories_text = "<b>📚 Available Categories</b>\n\n"
    for cat_type, count in sorted(categories.items()):
        categories_text += f"• <b>{cat_type.title()}</b>: {count} items\n"
    
    keyboard = []
    for cat_type in sorted(categories.keys()):
        keyboard.append([InlineKeyboardButton(
            f"🔍 {cat_type.title()} ({categories[cat_type]})",
            switch_inline_query_current_chat=f"!{cat_type} "
        )])
    
    keyboard.append([
        InlineKeyboardButton("🆘 Help", callback_data="help"),
        InlineKeyboardButton("⬅️ Back", callback_data="back_to_start")
    ])
    
    await update.callback_query.edit_message_text(
        categories_text,
        parse_mode='HTML',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def update_docs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Update documentation"""
    if update.callback_query:
        await update.callback_query.answer("Updating documentation...")
    
    message = await context.bot.send_message(
        update.effective_chat.id,
        "🔄 Updating documentation from Telegram...",
        parse_mode='HTML'
    )
    
    success = await search_system.update_documentation()
    
    if success:
        await context.bot.edit_message_text(
            f"✅ Documentation updated!\n📊 {len(search_system.full_cache['entities'])} entities loaded",
            update.effective_chat.id,
            message.message_id,
            parse_mode='HTML'
        )
    else:
        await context.bot.edit_message_text(
            "❌ Failed to update documentation",
            update.effective_chat.id,
            message.message_id,
            parse_mode='HTML'
        )

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle callback queries"""
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    if data == 'help':
        await help_command(update, context)
    elif data == 'categories':
        await show_categories(update, context)
    elif data == 'update_docs':
        await update_docs(update, context)
    elif data == 'back_to_start':
        await start(update, context)

async def inline_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline queries"""
    query = update.inline_query.query
    results = []
    
    if not query:
        # Show default suggestions
        results.extend(get_default_suggestions())
        await update.inline_query.answer(results, cache_time=0)
        return
    
    # Handle symbol formatting
    if '$' in query and not query.startswith('!') and not query.startswith('.'):
        formatted_text = search_system.format_symbols(query)
        sanitized_text = search_system.sanitize_html(formatted_text)
        
        results.append(InlineQueryResultArticle(
            id='format_symbols',
            title='🔤 Format API Symbols',
            description='Click to send formatted text with linked symbols',
            input_message_content=InputTextMessageContent(
                message_text=sanitized_text,
                parse_mode='HTML',
                disable_web_page_preview=True
            ),
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("🔍 Search Again", switch_inline_query_current_chat="")
            ]])
        ))
    
    # Search for entities
    search_results = search_system.search_entities(query, limit=50)
    
    if not search_results:
        results.append(InlineQueryResultArticle(
            id='no_results',
            title='😔 No Results Found',
            description='Try searching with different keywords',
            input_message_content=InputTextMessageContent(
                message_text=search_system.sanitize_html(
                    f"<b>❌ No matching documentation found for:</b>\n<code>{query}</code>\n\n"
                    f"🔁 Try again with a more specific keyword."
                ),
                parse_mode='HTML'
            ),
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("🔍 Search Again", switch_inline_query_current_chat="")
            ]])
        ))
    else:
        for item in search_results[:50]:
            # Build message text
            message_text = f"<b>{search_system.sanitize_html(item['name'])}</b>\n\n"
            
            if item.get('description_text'):
                desc = search_system.sanitize_html(item['description_text'])
                message_text += f"{desc}\n\n"
            
            # Add notes
            if item.get('notes'):
                message_text += "<b>📝 Notes:</b>\n"
                for note in item['notes']:
                    note_text = search_system.sanitize_html(note.get('content_text', ''))
                    message_text += f"• {note_text}\n"
                message_text += "\n"
            
            # Add fields
            if item.get('fields'):
                message_text += "<b>🔧 Fields/Parameters:</b>\n"
                for field in item['fields'][:10]:  # Limit fields
                    field_desc = search_system.sanitize_html(field.get('description_text', ''))
                    req = "✅" if field.get('required') else "❌"
                    message_text += f"• <code>{field['name']}</code> {req} - {field_desc}\n"
            
            # Truncate if too long
            if len(message_text) > 4000:
                message_text = search_system.truncate_safe_html(message_text, 4000)
            
            # Create result
            description = item.get('clean_desc', '')[:100] + '...' if len(item.get('clean_desc', '')) > 100 else item.get('clean_desc', '')
            
            results.append(InlineQueryResultArticle(
                id=str(item['id']),
                title=f"{item['name']} ({item['type']})",
                description=description,
                input_message_content=InputTextMessageContent(
                    message_text=message_text,
                    parse_mode='HTML',
                    disable_web_page_preview=True
                ),
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("📖 Open Docs", url=item['source_url'])],
                    [InlineKeyboardButton("🔍 Search Again", switch_inline_query_current_chat="")]
                ])
            ))
    
    await update.inline_query.answer(results, cache_time=0, is_personal=True)

def get_default_suggestions():
    """Get default inline suggestions"""
    return [
        InlineQueryResultArticle(
            id='welcome',
            title='👋 Welcome to TgDocsXBot!',
            description='Start typing to search Telegram Bot API documentation',
            input_message_content=InputTextMessageContent(
                message_text=(
                    "👋 <b>Welcome to TgDocsXBot!</b>\n\n"
                    "Use me inline to search Telegram Bot API documentation quickly!\n\n"
                    "Try: <code>@TgDocsXBot getUpdates</code>"
                ),
                parse_mode='HTML'
            ),
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("🔍 Start Searching", switch_inline_query_current_chat="")
            ]])
        ),
        InlineQueryResultArticle(
            id='examples',
            title='📚 Search Examples',
            description='Click to see search examples',
            input_message_content=InputTextMessageContent(
                message_text=(
                    "<b>🔍 Search Examples:</b>\n\n"
                    "<code>@TgDocsXBot getUpdates</code> - Basic search\n"
                    "<code>@TgDocsXBot !method send</code> - Only methods\n"
                    "<code>@TgDocsXBot Message .chat</code> - Message properties\n"
                    "<code>@TgDocsXBot $Message $Chat</code> - Format symbols\n\n"
                    "Start typing after @TgDocsXBot to search!"
                ),
                parse_mode='HTML'
            )
        )
    ]

async def main():
    """Start the bot"""
    # Initialize search system
    await search_system.init()
    
    # Create Application
    application = Application.builder().token("6703058288:AAGv4AAv4V9JnNWLTJ-PPtbjVAtGO8B6jBs").build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("update", update_docs))
    application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_handler(InlineQueryHandler(inline_query))
    
    # Start the Bot
    await application.run_polling()

if __name__ == '__main__':
    asyncio.run(main())