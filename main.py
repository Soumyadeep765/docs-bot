import os
import sqlite3
import logging
from flask import Flask, request, jsonify
from telegram import Update, InlineQueryResultArticle, InputTextMessageContent
from telegram.ext import Application, InlineQueryHandler, ContextTypes
import html
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
DB_FILE = 'bot_api.db'

BOT_TOKEN = "7828383323:AAFldn_KaEDMurNgC0pM-z4lAxd39UNpzNU"

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            content TEXT,
            description TEXT,
            clean_desc TEXT,
            last_updated TIMESTAMP,
            source_url TEXT NOT NULL
        )''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS fields (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            type TEXT,
            description TEXT,
            clean_desc TEXT,
            required INTEGER DEFAULT 0
        )''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type)')
        conn.commit()

init_db()

def sanitize_html(text):
    if not text:
        return ''
    allowed_tags = ['b', 'strong', 'i', 'em', 'u', 's', 'code', 'pre', 'a']
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
    def tag_handler(match):
        full_tag = match.group(0)
        tag_name = match.group(2).lower()
        if tag_name in allowed_tags:
            return full_tag
        return ''
    
    text = re.sub(r'<(/?)([a-zA-Z0-9]+)([^>]*)>', tag_handler, text)
    return text

def truncate_html(text, max_length):
    if len(text) <= max_length:
        return text
    
    open_tags = []
    truncated = ""
    pos = 0
    
    while pos < len(text) and len(truncated) < max_length:
        char = text[pos]
        if char == '<':
            tag_match = re.match(r'<(/)?([a-zA-Z]+)[^>]*>', text[pos:])
            if tag_match:
                tag_full = tag_match.group(0)
                is_closing = tag_match.group(1)
                tag_name = tag_match.group(2).lower()
                
                if is_closing:
                    if open_tags and open_tags[-1] == tag_name:
                        open_tags.pop()
                else:
                    open_tags.append(tag_name)
                
                truncated += tag_full
                pos += len(tag_full)
                continue
        
        truncated += char
        pos += 1
    
    truncated += "..."
    
    for tag in reversed(open_tags):
        truncated += f"</{tag}>"
    
    return truncated

async def handle_inline_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.inline_query.query.strip()
    
    if not query:
        await update.inline_query.answer(
            [],
            cache_time=1,
            is_personal=True,
            switch_pm_text="Enter a search term 🙂",
            switch_pm_parameter="start"
        )
        return
    
    try:
        with sqlite3.connect(DB_FILE) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT e.id, e.name, e.type, e.description, e.clean_desc, e.source_url
                FROM entities e
                WHERE LOWER(e.name) LIKE ? OR e.id IN (
                    SELECT entity_id FROM fields WHERE LOWER(name) LIKE ?
                )
                ORDER BY 
                    CASE WHEN LOWER(e.name) = ? THEN 1 
                         WHEN LOWER(e.name) LIKE ? THEN 2
                         ELSE 3 END,
                    e.name
                LIMIT 50
            ''', (f'%{query.lower()}%', f'%{query.lower()}%', query.lower(), f'{query.lower()}%'))
            
            results = []
            
            for row in cursor.fetchall():
                name = html.escape(row['name'])
                description = sanitize_html(row['description'] or '')
                clean_desc = html.escape(row['clean_desc'] or '')[:100]
                
                message_text = f"<b>{name}</b>\n\n{description}"
                
                if len(message_text) > 4096:
                    message_text = truncate_html(message_text, 4000) + "\n\n<i>More details available in full documentation</i>"
                
                results.append(InlineQueryResultArticle(
                    id=str(row['id']),
                    title=name,
                    description=clean_desc,
                    input_message_content=InputTextMessageContent(
                        message_text=message_text,
                        parse_mode='HTML',
                        disable_web_page_preview=True
                    ),
                    reply_markup={
                        'inline_keyboard': [[
                            {'text': '📖 Open Docs', 'url': row['source_url']},
                            {'text': '🔍 Search Again', 'switch_inline_query_current_chat': ''}
                        ]]
                    }
                ))
            
            if not results:
                results.append(InlineQueryResultArticle(
                    id="no_results",
                    title="😔 No Results Found",
                    description="Try searching with different keywords",
                    input_message_content=InputTextMessageContent(
                        message_text=f"<b>❌ No documentation found for:</b>\n<code>{html.escape(query)}</code>\n\n🔁 Try with different keywords.",
                        parse_mode='HTML'
                    ),
                    reply_markup={
                        'inline_keyboard': [[
                            {'text': '🔍 Search Again', 'switch_inline_query_current_chat': ''}
                        ]]
                    }
                ))
            
            await update.inline_query.answer(results, cache_time=1800, is_personal=False)
            
    except Exception as e:
        logger.error(f"Inline query failed: {str(e)}")
        await update.inline_query.answer([], cache_time=1)

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        json_data = request.get_json()
        update = Update.de_json(json_data, bot_app.bot)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(bot_app.process_update(update))
        loop.close()
        
        return jsonify({"status": "ok"})
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}")
        return jsonify({"status": "error"}), 500

@app.route('/api/search', methods=['GET'])
def api_search():
    query = request.args.get('q', '').strip()
    format_type = request.args.get('format', 'html')
    
    if not query:
        return jsonify({"error": "Missing query parameter"}), 400
    
    try:
        with sqlite3.connect(DB_FILE) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT e.id, e.name, e.type, e.description, e.clean_desc, e.source_url
                FROM entities e
                WHERE LOWER(e.name) LIKE ? OR e.id IN (
                    SELECT entity_id FROM fields WHERE LOWER(name) LIKE ?
                )
                ORDER BY 
                    CASE WHEN LOWER(e.name) = ? THEN 1 
                         WHEN LOWER(e.name) LIKE ? THEN 2
                         ELSE 3 END,
                    e.name
                LIMIT 20
            ''', (f'%{query.lower()}%', f'%{query.lower()}%', query.lower(), f'{query.lower()}%'))
            
            results = []
            for row in cursor.fetchall():
                result = {
                    'id': row['id'],
                    'name': row['name'],
                    'type': row['type'],
                    'description': row['description'] if format_type == 'html' else row['clean_desc'],
                    'reference': row['source_url']
                }
                
                cursor.execute('''
                    SELECT name, type, description, required
                    FROM fields 
                    WHERE entity_id = ?
                    ORDER BY required DESC, name
                ''', (row['id'],))
                
                fields = []
                for field_row in cursor.fetchall():
                    fields.append({
                        'name': field_row['name'],
                        'type': field_row['type'],
                        'description': field_row['description'],
                        'required': bool(field_row['required'])
                    })
                
                result['fields'] = fields
                results.append(result)
            
            return jsonify({
                'query': query,
                'count': len(results),
                'results': results,
                'format': format_type
            })
            
    except Exception as e:
        logger.error(f"API search failed: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    import asyncio
    
    bot_app = Application.builder().token(BOT_TOKEN).build()
    bot_app.add_handler(InlineQueryHandler(handle_inline_query))
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
