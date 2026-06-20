import os
import asyncio
import aiohttp
import numpy as np
import re
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm.ollama import ollama_model_complete

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
    """Читает книгу из файла и разбивает на главы по маркеру Глава X"""
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

# --- ОСНОВНАЯ ФУНКЦИЯ ---
async def main():
    # Удаляем всё, что связано с OpenAI
    for key in list(os.environ.keys()):
        if "OPENAI" in key:
            os.environ.pop(key, None)
    
    # --- ИНИЦИАЛИЗАЦИЯ LIGHTRAG ---
    rag = LightRAG(
        working_dir="./test_storage",
        llm_model_func=ollama_model_complete,
        embedding_func=ollama_embed,
        llm_model_name="mistral",
    )

    await rag.initialize_storages()
    
    # --- ЧИТАЕМ КНИГУ ПО ГЛАВАМ ---
    book_file = "book.txt"
    if not os.path.exists(book_file):
        print(f"❌ Файл {book_file} не найден. Создайте его с главами.")
        return
    
    chapters = read_book_by_chapters(book_file)
    print(f"📖 Загружено {len(chapters)} глав")
    
    # --- ВЫБОР ТЕКУЩЕЙ ГЛАВЫ ---
    current_chapter = 1
    print(f"\n📖 Текущая глава: {current_chapter}")
    print("Чтобы изменить, напишите /chapter N")
    
    # --- ИНДЕКСАЦИЯ ТОЛЬКО ПРОЧИТАННЫХ ГЛАВ ---
    read_text = ""
    for i in range(1, current_chapter + 1):
        if i in chapters:
            read_text += f"Глава {i}\n{chapters[i]}\n\n"
    
    await rag.ainsert(read_text)
    print(f"\n✅ Проиндексировано глав: {current_chapter}")
    print("\n📚 Задавайте вопросы по прочитанным главам (для выхода напишите 'выход'):")

    # --- ЦИКЛ ВВОДА ВОПРОСОВ ---
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
                    read_text = ""
                    for i in range(1, current_chapter + 1):
                        if i in chapters:
                            read_text += f"Глава {i}\n{chapters[i]}\n\n"
                    await rag.ainsert(read_text)
                    print(f"✅ Переключился на главу {current_chapter}")
                else:
                    print(f"⚠️ Глава {new_chapter} не найдена. Доступно глав: {len(chapters)}")
            except ValueError:
                print("⚠️ Используй формат: /chapter N")
            continue
        
        if user_question.strip():
            print("⏳ Думаю...")
            
            # --- УЛУЧШЕННЫЙ ПРОМПТ ---
            enhanced_question = (
    f"Пользователь: {user_question}\n\n"
    f"Ты — собеседник по книге. Прочитано до главы {current_chapter}.\n\n"
    f"Твоя задача — вести живой диалог, а не просто давать факты.\n\n"
    f"Правила:\n"
    f"1. Если пользователь спрашивает факт — ответь кратко и задай уточняющий вопрос.\n"
    f"2. Если пользователь высказывает мнение — согласись или мягко поспорь, аргументируя по тексту.\n"
    f"3. Если пользователь спрашивает «почему» — предложи свой анализ персонажа или ситуации.\n"
    f"4. Если пользователь просит твоё мнение — выскажи его, основываясь на прочитанном.\n"
    f"5. Если ответа нет в прочитанных главах — скажи честно: 'Ответа нет в прочитанных главах'.\n"
    f"6. Если в вопросе есть слово, которого нет в книге, объясни его значение из своих знаний языка.\n"
    f"7. Если слово есть в книге, объясни его по тексту.\n"
    f"8. НЕ упоминай события из глав {current_chapter + 1} и дальше — это спойлеры.\n"
    f"9. Отвечай на русском языке, в формате живого диалога."
    f"10. Если полtelegram_bot.pyьзователь написал 'привет' или 'здравствуй' — поздоровайся и спроси, что он хочет обсудить."
    f"11. Если пользователь не написал 'привет' или 'здравствуй' (НЕ ПОЗДОРОВАЛСЯ), то НЕ НУЖНО здороваться в ответ."
    f"12. Точно цитируй текст, если это важно."
)
            
            full_result = await rag.aquery(
                enhanced_question,
                param=QueryParam(mode="hybrid")
            )
            
            # Берём весь ответ
            short_answer = full_result.strip()
            
            print(f"\n📖 Ответ LightRAG:\n{short_answer}\n")
        else:
            print("⚠️ Пожалуйста, введите вопрос.")

if __name__ == "__main__":
    asyncio.run(main())