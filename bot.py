import os
import asyncio
import aiohttp
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm.ollama import ollama_model_complete

# --- ФУНКЦИЯ ДЛЯ ЭМБЕДДИНГОВ ЧЕРЕЗ OLLAMA ---
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

# --- ОСНОВНАЯ ФУНКЦИЯ ---
async def main():
    # Удаляем всё, что связано с OpenAI
    for key in list(os.environ.keys()):
        if "OPENAI" in key:
            os.environ.pop(key, None)
    
    # --- ИНИЦИАЛИЗАЦИЯ LIGHTRAG С OLLAMA ---
    rag = LightRAG(
        working_dir="./test_storage",
        llm_model_func=ollama_model_complete,
        embedding_func=ollama_embed,
        llm_model_name="llama3.1",
    )

    await rag.initialize_storages()
    
    # Создаём тестовый файл, если его нет
    if not os.path.exists("test.txt"):
        with open("test.txt", "w", encoding="utf-8") as f:
            f.write("Это тестовая книга. Главный герой — кот по имени Барсик. Он любит молоко и спит на подоконнике. Однажды Барсик нашёл старую карту сокровищ и отправился в путешествие.")

    with open("test.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # --- ИНДЕКСАЦИЯ И ЗАПРОС ---
    await rag.ainsert(text)
    
    # Получаем полный ответ
    full_result = await rag.aquery("Что любит Барсик?", param=QueryParam(mode="hybrid"))
    
    # Берём только первую строку (сам ответ)
    short_answer = full_result.split('\n')[0]
    
    print("\nОтвет LightRAG (короткий):")
    print(short_answer)

if __name__ == "__main__":
    asyncio.run(main())