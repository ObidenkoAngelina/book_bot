import os
import asyncio
import aiohttp
import numpy as np
import re
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm.ollama import ollama_model_complete

# ==================== КОНФИГУРАЦИЯ ====================
TELEGRAM_TOKEN = "8863485135:AAE4xZorGJXS3odOnYXcaCR3tcYM-asbTP8"
WORKING_DIR = "./telegram_storage"
BOOKS_DIR = "./books"
GRAPHS_DIR = "./graphs"

# Список доступных книг (заранее проиндексированных)
AVAILABLE_BOOKS = {
    "little_prince": {
        "title": "Маленький принц",
        "author": "Антуан де Сент-Экзюпери",
        "chapters": 27,
        "graph_dir": f"{GRAPHS_DIR}/little_prince",
    },
    "little_prince2": {
        "title": "Маленький принц2",
        "author": "Антуан де Сент-Экзюпери",
        "chapters": 27,
        "graph_dir": f"{GRAPHS_DIR}/little_prince2",
    },
}

# ==================== ФУНКЦИЯ ДЛЯ ЭМБЕДДИНГОВ ====================
async def ollama_embed_fn(texts: list[str]) -> np.ndarray:
    url = "http://localhost:11434/api/embeddings"
    embeddings = []
    for text in texts:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json={"model": "nomic-embed-text", "prompt": text}) as response:
                    if response.status == 200:
                        result = await response.json()
                        embeddings.append(result["embedding"])
                    else:
                        embeddings.append([0.0] * 768)
        except Exception:
            embeddings.append([0.0] * 768)
    return np.array(embeddings, dtype=np.float32)

ollama_embed = EmbeddingFunc(
    embedding_dim=768,
    max_token_size=8192,
    func=ollama_embed_fn
)

# ==================== ЗАГРУЗКА ГРАФА (АСИНХРОННАЯ) ====================
async def load_graph(book_name: str, chapter_num: int):
    """Загружает сохранённый граф для указанной книги и главы"""
    path = f"{GRAPHS_DIR}/{book_name}/ch_{chapter_num:02d}"
    
    if not os.path.exists(path):
        print(f"❌ Папка не найдена: {path}")
        return None
    
    rag = LightRAG(
        working_dir=path,
        llm_model_func=ollama_model_complete,
        embedding_func=ollama_embed,
        llm_model_name="mistral",
    )
    await rag.initialize_storages()
    print(f"✅ Граф загружен: {path}")
    return rag

# ==================== КОМАНДЫ БОТА ====================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает список доступных книг"""
    keyboard = []
    for key, book in AVAILABLE_BOOKS.items():
        keyboard.append([InlineKeyboardButton(
            f"📖 {book['title']} — {book['author']}",
            callback_data=f"select_book_{key}"
        )])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "📚 Привет! Я книжный собеседник.\n\n"
        "Выбери книгу из списка:",
        reply_markup=reply_markup
    )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка нажатий на кнопки"""
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    if data.startswith("select_book_"):
        book_key = data.replace("select_book_", "")
        book_info = AVAILABLE_BOOKS[book_key]
        
        context.user_data['book_key'] = book_key
        context.user_data['book_title'] = book_info['title']
        context.user_data['current_chapter'] = 1
        context.user_data['total_chapters'] = book_info['chapters']
        
        # Загружаем граф для первой главы (АСИНХРОННО)
        rag = await load_graph(book_key, 1)
        if rag is None:
            await query.edit_message_text(
                f"❌ Граф для книги '{book_info['title']}' не найден.\n"
                f"Сначала проиндексируйте книгу локально."
            )
            return
        
        context.user_data['rag'] = rag
        
        # Клавиатура для управления
        keyboard = [
            [InlineKeyboardButton("📖 Моя книга", callback_data="mybook")],
            [InlineKeyboardButton("📍 Сменить главу", callback_data="change_chapter")],
            [InlineKeyboardButton("📚 Выбрать другую книгу", callback_data="back_to_books")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            f"✅ Выбрана книга: **{book_info['title']}**\n"
            f"📖 Всего глав: {book_info['chapters']}\n"
            f"📍 Ты на главе: 1\n\n"
            f"Теперь задавай вопросы по тексту!",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
    
    elif data == "mybook":
        book_title = context.user_data.get('book_title', 'Не выбрана')
        chapter = context.user_data.get('current_chapter', 1)
        total = context.user_data.get('total_chapters', '?')
        await query.edit_message_text(
            f"📖 Ты читаешь: **{book_title}**\n"
            f"📍 Дочитал до главы: {chapter} из {total}",
            parse_mode="Markdown"
        )
        await show_main_menu(query, context)
    
    elif data == "change_chapter":
        total = context.user_data.get('total_chapters', 27)
        keyboard = []
        row = []
        for i in range(1, min(total + 1, 31)):
            row.append(InlineKeyboardButton(str(i), callback_data=f"set_chapter_{i}"))
            if len(row) == 5:
                keyboard.append(row)
                row = []
        if row:
            keyboard.append(row)
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            "📍 Выбери главу, до которой дочитал:",
            reply_markup=reply_markup
        )
    
    elif data.startswith("set_chapter_"):
        chapter = int(data.replace("set_chapter_", ""))
        book_key = context.user_data.get('book_key')
        book_title = context.user_data.get('book_title')
        
        if not book_key:
            await query.edit_message_text("❌ Сначала выбери книгу.")
            return
        
        context.user_data['current_chapter'] = chapter
        
        # Загружаем граф для новой главы (АСИНХРОННО)
        rag = await load_graph(book_key, chapter)
        if rag is None:
            await query.edit_message_text(
                f"❌ Граф для главы {chapter} не найден.\n"
                f"Убедитесь, что книга проиндексирована до этой главы."
            )
            return
        
        context.user_data['rag'] = rag
        
        await query.edit_message_text(
            f"✅ Переключился на главу {chapter} книги **{book_title}**\n\n"
            f"Теперь я не знаю, что будет в следующих главах!",
            parse_mode="Markdown"
        )
        await show_main_menu(query, context)
    
    elif data == "back_to_books":
        await start(update, context)
    
    elif data == "back_to_menu":
        await show_main_menu(query, context)

async def show_main_menu(query, context):
    """Показывает главное меню"""
    keyboard = [
        [InlineKeyboardButton("📖 Моя книга", callback_data="mybook")],
        [InlineKeyboardButton("📍 Сменить главу", callback_data="change_chapter")],
        [InlineKeyboardButton("📚 Выбрать другую книгу", callback_data="back_to_books")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        f"📖 Ты читаешь: **{context.user_data.get('book_title', 'Не выбрана')}**\n"
        f"📍 Глава: {context.user_data.get('current_chapter', 1)}\n\n"
        f"Задавай вопросы!",
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка вопросов пользователя"""
    question = update.message.text
    
    if 'rag' not in context.user_data:
        await update.message.reply_text("Сначала выбери книгу командой /start")
        return
    
    rag = context.user_data['rag']
    current_chapter = context.user_data.get('current_chapter', 1)
    book_title = context.user_data.get('book_title', '')
    
    await update.message.reply_text("⏳ Думаю...")
    
    try:
        enhanced_question = (
            f"Пользователь: {question}\n\n"
            f"Ты — собеседник по книге «{book_title}». Прочитано до главы {current_chapter}.\n\n"
            f"Твоя задача — отвечать на вопросы ТОЛЬКО по тексту книги.\n\n"
            f"Правила:\n"
            f"1. Дай ПРЯМОЙ ответ на вопрос по тексту (кратко, 1-5 предложений).\n"
            f"2. Если ответа нет в прочитанных главах — скажи честно: 'В прочитанных главах этого нет'. НЕ ВЫДУМЫВАЙ.\n"
            f"3. НЕ выдумывай детали, имена, события, которых нет в тексте.\n"
            f"4. Если в тексте есть цитата — процитируй её.\n"
            f"5. НЕ упоминай события из глав {current_chapter + 1} и дальше — это спойлеры.\n"
            f"6. Отвечай на русском языке, кратко и по делу."
        )
        
        full_result = await rag.aquery(
            enhanced_question,
            param=QueryParam(mode="hybrid")
        )
        
        if full_result is None:
            answer = "Извините, не удалось найти ответ на ваш вопрос в прочитанных главах."
        else:
            answer = full_result.strip()
            if len(answer) > 4000:
                answer = answer[:4000] + "..."
        
        await update.message.reply_text(f"📖 {answer}")
        
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")

# ==================== ЗАПУСК БОТА ====================
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))
    
    print("Telegram-бот запущен...")
    app.run_polling()

if __name__ == "__main__":
    main()