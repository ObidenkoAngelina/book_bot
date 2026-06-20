import os
import asyncio
import aiohttp
import numpy as np
import re
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm.ollama import ollama_model_complete

# --- НАСТРОЙКИ ---
TELEGRAM_TOKEN = "8863485135:AAE4xZorGJXS3odOnYXcaCR3tcYM-asbTP8"  # Вставьте сюда!
WORKING_DIR = "./telegram_storage"

# --- ФУНКЦИЯ ДЛЯ ЭМБЕДДИНГОВ ---
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

# --- ФУНКЦИЯ ДЛЯ ЧТЕНИЯ КНИГИ ПО ГЛАВАМ ---
def read_book_by_chapters(content: str) -> dict:
    """Разбивает текст на главы по маркеру Глава X"""
    chapters = {}
    parts = re.split(r'(Глава \d+)', content)
    
    for i in range(1, len(parts), 2):
        chapter_title = parts[i].strip()
        chapter_content = parts[i+1].strip() if i+1 < len(parts) else ""
        chapter_num = int(re.search(r'\d+', chapter_title).group())
        chapters[chapter_num] = chapter_content
    
    return chapters

# --- ФУНКЦИЯ ДЛЯ ПОЛУЧЕНИЯ RAG (ДЛЯ КАЖДОГО ПОЛЬЗОВАТЕЛЯ) ---
def get_rag(user_id: int, book_name: str):
    """Создаёт или подключает RAG для конкретной книги пользователя"""
    user_dir = os.path.join(WORKING_DIR, f"user_{user_id}_{book_name}")
    os.makedirs(user_dir, exist_ok=True)
    
    rag = LightRAG(
        working_dir=user_dir,
        llm_model_func=ollama_model_complete,
        embedding_func=ollama_embed,
        llm_model_name="mistral",
    )
    return rag

# --- ОБРАБОТЧИКИ КОМАНД ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📚 Привет! Я книжный собеседник Кейн.\n\n"
        "Загрузи книгу в формате TXT, и я буду обсуждать её с тобой.\n"
        "Я не спойлерю — я знаю, на какой главе ты остановился.\n\n"
        "📖 Команды:\n"
        "/start — показать это сообщение\n"
        "/chapter N — указать, что ты дочитал до главы N\n"
        "/mybook — показать текущую книгу и главу"
    )

async def set_chapter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Пользователь указывает текущую главу"""
    if 'current_book' not in context.user_data:
        await update.message.reply_text("Сначала загрузи книгу.")
        return
    
    try:
        chapter = int(context.args[0])
        context.user_data['current_chapter'] = chapter
        
        # Переиндексируем книгу до новой главы
        user_id = update.message.from_user.id
        book_name = context.user_data['current_book']
        book_content = context.user_data.get('book_content', '')
        
        chapters = read_book_by_chapters(book_content)
        read_text = ""
        for i in range(1, chapter + 1):
            if i in chapters:
                read_text += f"Глава {i}\n{chapters[i]}\n\n"
        
        rag = get_rag(user_id, book_name)
        await rag.initialize_storages()
        await rag.ainsert(read_text)
        
        await update.message.reply_text(f"📖 Понял! Ты дочитал(а) до главы {chapter}. Я не буду забегать вперёд!")
    except (IndexError, ValueError):
        await update.message.reply_text("Напиши: /chapter N, где N — номер главы")

async def mybook(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает текущую книгу и главу"""
    if 'current_book' not in context.user_data:
        await update.message.reply_text("Ты ещё не загрузил книгу.")
        return
    
    book_name = context.user_data['current_book']
    chapter = context.user_data.get('current_chapter', 1)
    await update.message.reply_text(f"📖 Ты читаешь: {book_name}\n📍 Дочитал до главы: {chapter}")

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка загруженного файла"""
    user_id = update.message.from_user.id
    document = update.message.document
    
    # Проверяем, что это TXT
    if not document.file_name.endswith('.txt'):
        await update.message.reply_text("Пожалуйста, загрузи файл в формате .txt")
        return
    
    # Скачиваем файл
    file = await context.bot.get_file(document.file_id)
    book_name = document.file_name.replace('.txt', '')
    file_content = await file.download_as_bytearray()
    book_text = file_content.decode('utf-8', errors='ignore')
    
    await update.message.reply_text(f"📖 Книга '{document.file_name}' загружается. Это может занять некоторое время...")
    
    try:
        # --- ИНДЕКСАЦИЯ КНИГИ ---
        chapters = read_book_by_chapters(book_text)
        
        if not chapters:
            await update.message.reply_text("❌ Не удалось найти главы в книге. Убедитесь, что они отмечены как 'Глава 1', 'Глава 2' и т.д.")
            return
        
        # Индексируем только первую главу (по умолчанию)
        current_chapter = 1
        read_text = ""
        for i in range(1, current_chapter + 1):
            if i in chapters:
                read_text += f"Глава {i}\n{chapters[i]}\n\n"
        
        rag = get_rag(user_id, book_name)
        await rag.initialize_storages()
        await rag.ainsert(read_text)
        
        # Сохраняем информацию о книге
        context.user_data['current_book'] = book_name
        context.user_data['current_chapter'] = current_chapter
        context.user_data['book_content'] = book_text
        
        await update.message.reply_text(
            f"✅ Книга '{document.file_name}' успешно загружена!\n"
            f"📖 Найдено глав: {len(chapters)}\n"
            f"📍 Ты на главе 1\n\n"
            f"Теперь задавай вопросы. Чтобы переключить главу, используй /chapter N"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка при загрузке книги: {e}")

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка вопросов пользователя"""
    user_id = update.message.from_user.id
    question = update.message.text
    
    # Проверяем, есть ли загруженная книга
    if 'current_book' not in context.user_data:
        await update.message.reply_text("Сначала загрузи книгу командой /start и отправь файл.")
        return
    
    book_name = context.user_data['current_book']
    current_chapter = context.user_data.get('current_chapter', 1)
    book_content = context.user_data.get('book_content', '')
    
    await update.message.reply_text("⏳ Думаю...")
    
    try:
        # --- ФОРМИРУЕМ ПРОМПТ (как в консольном боте) ---
        enhanced_question = (
            f"Пользователь: {question}\n\n"
            f"Ты — собеседник по книге. Прочитано до главы {current_chapter}.\n\n"
            f"Твоя задача — вести живой диалог, а не просто давать факты.\n\n"
            f"Правила:\n"
            f"1. Если пользователь спрашивает факт — ответь кратко (1-5 предложений) и задай уточняющий вопрос.\n"
            f"2. Если пользователь высказывает мнение — согласись или мягко поспорь, аргументируя по тексту.\n"
            f"3. Если пользователь спрашивает «почему» — предложи свой анализ персонажа или ситуации.\n"
            f"4. Если пользователь просит твоё мнение — выскажи его, основываясь на прочитанном.\n"
            f"5. Если ответа нет в прочитанных главах — скажи честно: 'Ответа нет в прочитанных главах'.\n"
            f"6. Если в вопросе есть слово, которого нет в книге, объясни его значение из своих знаний языка.\n"
            f"7. Если слово есть в книге, объясни его по тексту.\n"
            f"8. НЕ упоминай события из глав {current_chapter + 1} и дальше — это спойлеры.\n"
            f"9. Отвечай на русском языке, в формате живого диалога.\n"
            f"10. Если пользователь написал 'привет' или 'здравствуй' — поздоровайся и спроси, что он хочет обсудить.\n"
            f"11. Если пользователь не написал 'привет' или 'здравствуй' — НЕ НУЖНО здороваться в ответ.\n"
        )
        
        # Получаем RAG для этого пользователя
        rag = get_rag(user_id, book_name)
        await rag.initialize_storages()
        
        # Запрос
        full_result = await rag.aquery(
            enhanced_question,
            param=QueryParam(mode="hybrid")
        )
        
        # Ответ
        answer = full_result.strip()
        
        # Если ответ слишком длинный, обрезаем (Telegram не любит очень длинные сообщения)
        if len(answer) > 4000:
            answer = answer[:4000] + "..."
        
        await update.message.reply_text(f"📖 {answer}")
        
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")

# --- ЗАПУСК БОТА ---
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("chapter", set_chapter))
    app.add_handler(CommandHandler("mybook", mybook))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))
    
    print("🤖 Telegram-бот 'Порфирьевич' запущен...")
    app.run_polling()

if __name__ == "__main__":
    main()