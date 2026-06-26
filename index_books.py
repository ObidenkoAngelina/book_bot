import os
import asyncio
import aiohttp
import numpy as np
import re
import shutil
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm.ollama import ollama_model_complete

# --- КОНФИГУРАЦИЯ ---
BOOK_NAME = "little_prince2"
BOOK_FILE = f"books/{BOOK_NAME}/full_text.txt"
GRAPHS_DIR = f"graphs/{BOOK_NAME}"
TEMP_DIR = f"{GRAPHS_DIR}/temp"

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
def read_book_by_chapters(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    chapters = {}
    parts = re.split(r'(Глава \d+)', content)
    
    for i in range(1, len(parts), 2):
        chapter_title = parts[i].strip()
        chapter_content = parts[i+1].strip() if i+1 < len(parts) else ""
        chapter_num = int(re.search(r'\d+', chapter_title).group())
        chapters[chapter_num] = chapter_content
    
    return chapters

# --- ФУНКЦИИ ДЛЯ РАБОТЫ С ГРАФАМИ (ЧЕРЕЗ КОПИРОВАНИЕ ПАПКИ) ---
def save_graph_state(chapter_num: int):
    src = TEMP_DIR
    dst = f"{GRAPHS_DIR}/ch_{chapter_num:02d}"
    
    if not os.path.exists(src):
        print(f"❌ Папка {src} не найдена")
        return
    
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"✅ Граф для главы {chapter_num} сохранён")

async def load_graph(chapter_num: int):
    path = f"{GRAPHS_DIR}/ch_{chapter_num:02d}"
    if not os.path.exists(path):
        return None
    
    rag = LightRAG(
        working_dir=path,
        llm_model_func=ollama_model_complete,
        embedding_func=ollama_embed,
        llm_model_name="qwen2.5:3b",   # ← ИСПРАВЛЕНО
        )
    await rag.initialize_storages()
    return rag

def graph_exists(chapter_num: int) -> bool:
    return os.path.exists(f"{GRAPHS_DIR}/ch_{chapter_num:02d}")

# --- ИНДЕКСАЦИЯ ---
async def index_all_chapters(chapters: dict):
    print("🚀 Начинаю индексацию всех глав через qwen2.5:3b...")
    
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    
    rag = LightRAG(
        working_dir=TEMP_DIR,
        llm_model_func=ollama_model_complete,
        embedding_func=ollama_embed,
        llm_model_name="qwen2.5:3b",   # ← ИСПРАВЛЕНО
    )
    await rag.initialize_storages()
    
    for chapter_num in sorted(chapters.keys()):
        chapter_text = chapters[chapter_num]
        print(f"📖 Индексация главы {chapter_num}...")
        await rag.ainsert(chapter_text)
        save_graph_state(chapter_num)
    
    print(f"✅ Индексация завершена! Сохранено {len(chapters)} графов.")

# --- ОСНОВНАЯ ФУНКЦИЯ ---
async def main():
    for key in list(os.environ.keys()):
        if "OPENAI" in key:
            os.environ.pop(key, None)
    
    if not os.path.exists(BOOK_FILE):
        print(f"❌ Файл {BOOK_FILE} не найден.")
        return
    
    chapters = read_book_by_chapters(BOOK_FILE)
    print(f"📖 Загружено {len(chapters)} глав")
    
    if not graph_exists(1):
        print("⚠️ Графы не найдены. Запускаю индексацию...")
        await index_all_chapters(chapters)
    else:
        print("✅ Графы уже существуют, загружаю...")
    
    current_chapter = 1
    rag = await load_graph(current_chapter)
    
    if rag is None:
        print("❌ Не удалось загрузить граф.")
        return
    
    print(f"\n📖 Текущая глава: {current_chapter}")
    print("Чтобы изменить, напишите /chapter N")
    print("📚 Задавайте вопросы (для выхода напишите 'выход'):")
    
    while True:
        user_question = input("\n❓ Ваш вопрос: ")
        
        if user_question.lower() in ['выход', 'exit', 'quit', 'q']:
            print("👋 До свидания!")
            break
        
        if user_question.startswith('/chapter'):
            try:
                new_chapter = int(user_question.split()[1])
                if 1 <= new_chapter <= len(chapters):
                    current_chapter = new_chapter
                    rag = await load_graph(current_chapter)
                    if rag is None:
                        print(f"❌ Граф для главы {current_chapter} не найден.")
                        continue
                    print(f"✅ Переключился на главу {current_chapter}")
                else:
                    print(f"⚠️ Глава {new_chapter} не найдена. Доступно: 1-{len(chapters)}")
            except ValueError:
                print("⚠️ Используй формат: /chapter N")
            continue
        
        if user_question.strip():
            print("⏳ Думаю...")
            
            enhanced_question = (
    f"Пользователь: {user_question}\n\n"
    f"Ты — собеседник по книге «Маленький принц». Прочитано до главы {current_chapter}.\n\n"
    f"Твоя задача — отвечать на вопросы ТОЛЬКО по тексту книги. Не используй свои знания о мире, только то, что есть в книге.\n\n"
    f"Правила:\n"
    f"1. Если в прочитанных главах есть точный ответ — дай его ПРЯМО и КРАТКО.\n"
    f"2. Если в прочитанных главах нет точного ответа — скажи: 'В прочитанных главах этого нет'.\n"
    f"3. НЕ выдумывай детали, имена, события, которых нет в тексте.\n"
    f"4. Если в тексте есть цитата, которая отвечает на вопрос — процитируй её ДОСЛОВНО.\n"
    f"5. НЕ упоминай события из глав {current_chapter + 1} и дальше — это спойлеры.\n"
    f"6. Отвечай на русском языке, кратко и по делу.\n"
    f"7. Если вопрос про персонажа — назови его имя и что он делает в книге.\n"
    f"8. Если вопрос про событие — опиши его, но только по тексту.\n"
    f"9. Если вопрос общий ('что такое', 'кто такой') — дай определение из книги.\n"
    f"10. Если не знаешь — скажи честно, не придумывай."
)
            
            full_result = await rag.aquery(
                enhanced_question,
                param=QueryParam(
                    mode="naive",
                    chunk_top_k=100
                )
            )
            
            print(f"\n📖 Ответ:\n{full_result.strip()}\n")
        else:
            print("⚠️ Введите вопрос.")

if __name__ == "__main__":
    asyncio.run(main())