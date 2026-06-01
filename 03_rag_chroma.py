"""
03_rag_chroma.py — RAG с настоящей векторной БД (ChromaDB)
═══════════════════════════════════════════════════════════════════
Цель: показать как ровно тот же RAG-пайплайн выглядит при
использовании промышленной векторной базы данных.

Что нового по сравнению с 02_naive_rag.py:
  • Векторы хранятся на диске — не нужно переиндексировать при перезапуске.
  • Поиск работает через ANN (Approximate Nearest Neighbors).
  • Каждый чанк имеет МЕТАДАННЫЕ: имя источника, индекс, длина.
  • Можно индексировать НЕСКОЛЬКО документов и фильтровать по источнику.

Команды CLI:
  index <путь_к_файлу>   — добавить документ в индекс
  ask <вопрос>           — задать вопрос
  list                   — показать загруженные документы
  clear                  — очистить всю БД
  quit                   — выход

Требования:
  - LM Studio запущен (порт 1234)
  - Загружены: embedding-модель + LLM
  - pip install openai chromadb

Запуск: python 03_rag_chroma.py
"""

import os
import shlex
import chromadb
from openai import OpenAI

# ********************* КОНФИГУРАЦИЯ *********************

LM_STUDIO_URL   = "http://127.0.0.1:1234/v1"
EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"
LLM_MODEL       = "google/gemma-4-e4b"

# Папка где Chroma будет хранить базу (создаётся автоматически)
CHROMA_DIR     = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION     = "rag_lecture_11"   # имя коллекции внутри Chroma

# Параметры чанкинга и поиска (те же что и в 02)
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100
TOP_K         = 3

MAX_TOKENS  = 512
TEMPERATURE = 0.3


# ********************* ИНИЦИАЛИЗАЦИЯ *********************

# OpenAI-клиент для эмбеддингов и LLM
client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")

# ChromaDB-клиент с persistent storage:
# при первом запуске создаст папку и схему, дальше — переиспользует
chroma = chromadb.PersistentClient(path=CHROMA_DIR)

# Коллекция — аналог «таблицы». Если уже существует — открываем её,
# иначе создаём новую. metadata cosine = используем косинусное сходство.
collection = chroma.get_or_create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"},   # тип расстояния: cosine, l2, ip
)


# ********************* ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ *********************

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Скользящее окно с перекрытием — то же что в 02_naive_rag.py."""
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


def embed_text(text: str) -> list[float]:
    """Эмбеддинг через LM Studio. Возвращает list[float] — нативный формат для Chroma."""
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Батчевый эмбеддинг — отправляем сразу список текстов, экономим запросы."""
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    # data — список объектов с .embedding в том же порядке что и input
    return [d.embedding for d in resp.data]


# ********************* КОМАНДА: index <файл> *********************

def cmd_index(filepath: str):
    """Загружает документ, чанкует, считает эмбеддинги, кладёт в Chroma."""
    if not os.path.exists(filepath):
        print(f"  ❌ Файл не найден: {filepath}")
        return

    # 1. Читаем файл
    with open(filepath, encoding="utf-8") as f:
        text = f.read()
    source = os.path.basename(filepath)
    print(f"  📄 Прочитано {len(text)} символов из {source}")

    # 2. Дробим на чанки
    chunks = chunk_text(text)
    print(f"  ✂️  Получено {len(chunks)} чанков (L={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # 3. Считаем эмбеддинги (батчем — быстрее чем по одному)
    print(f"  🧮 Считаю эмбеддинги через {EMBEDDING_MODEL}...")
    embeddings = embed_batch(chunks)
    print(f"      Размерность: {len(embeddings[0])}")

    # 4. Готовим данные для Chroma
    # ids       — уникальные идентификаторы каждой записи (string)
    # documents — сами тексты чанков (Chroma вернёт их при поиске)
    # metadatas — любая дополнительная информация (источник, индекс, длина)
    # embeddings — заранее посчитанные векторы (передаём явно — иначе Chroma
    #              попыталась бы посчитать их сама своей моделью)
    ids       = [f"{source}::{i}" for i in range(len(chunks))]
    metadatas = [{"source": source, "chunk_index": i, "length": len(c)}
                 for i, c in enumerate(chunks)]

    # 5. Добавляем в коллекцию (upsert — обновит если ID уже существует)
    collection.upsert(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    print(f"  ✅ Добавлено в БД. Всего записей в коллекции: {collection.count()}")


# ********************* КОМАНДА: ask <вопрос> *********************

PROMPT_TEMPLATE = """Ты — ассистент, отвечающий на вопросы СТРОГО на основе предоставленного контекста.
Если в контексте нет ответа — честно скажи «В предоставленных документах ответа нет».
Не выдумывай факты. Не используй внешние знания.

КОНТЕКСТ:
{context}

ВОПРОС: {question}

ОТВЕТ:"""


def cmd_ask(question: str):
    """Полный цикл: вопрос → поиск в Chroma → промпт → LLM → ответ."""

    if collection.count() == 0:
        print("  ⚠️  База пуста. Сначала проиндексируйте документ:")
        print("      index sample_doc.txt")
        return

    # 1. Эмбеддинг вопроса
    q_vec = embed_text(question)

    # 2. Поиск top-K в Chroma. Передаём ВЕКТОР (не текст!) — это важно:
    #    если передать query_texts, Chroma попыталась бы посчитать эмбеддинг
    #    своей default-моделью, а нам нужна именно та модель из LM Studio.
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=TOP_K,
        # Можно фильтровать по метаданным, например только из конкретного источника:
        # where={"source": "sample_doc.txt"},
    )

    # Распаковываем (Chroma возвращает списки списков — для нескольких запросов сразу)
    docs       = results["documents"][0]    # тексты чанков
    metas      = results["metadatas"][0]    # метаданные
    distances  = results["distances"][0]    # расстояния (1 - cosine)

    # 3. Собираем контекст с понятными метками источников
    context_parts = []
    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
        # При cosine-метрике Chroma возвращает (1 - cosine_similarity)
        similarity = 1.0 - dist
        context_parts.append(
            f"[Источник: {meta['source']}, фрагмент #{meta['chunk_index']}, "
            f"похожесть={similarity:.2f}]\n{doc}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # 4. Формируем промпт и зовём LLM
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    answer = resp.choices[0].message.content or "[пустой ответ]"

    # 5. Показываем источники для прозрачности
    print("\n📚 Использованные фрагменты:")
    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
        preview = doc[:90].replace("\n", " ")
        print(f"  #{rank}  {meta['source']}  чанк {meta['chunk_index']}  "
              f"cosine={1-dist:.3f}  «{preview}...»")

    print(f"\n💬 Ответ:\n{answer}\n")


# ********************* ВСПОМОГАТЕЛЬНЫЕ КОМАНДЫ *********************

def cmd_list():
    """Показывает какие документы загружены в БД."""
    n = collection.count()
    if n == 0:
        print("  📭 База пуста.")
        return

    # peek(n) — возвращает первые n записей (без поиска, просто посмотреть)
    sample = collection.peek(limit=n)
    sources = {}
    for meta in sample["metadatas"]:
        sources[meta["source"]] = sources.get(meta["source"], 0) + 1

    print(f"  📊 Всего записей: {n}")
    print(f"  📚 Документов:")
    for src, count in sources.items():
        print(f"      • {src}  ({count} чанков)")


def cmd_clear():
    """Удаляет всю коллекцию и создаёт заново — простой способ всё стереть."""
    global collection
    chroma.delete_collection(COLLECTION)
    collection = chroma.get_or_create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    print("  🗑️  База очищена.")


# ********************* ТОЧКА ВХОДА: CLI *********************

HELP = """
Команды:
  index <файл>      — добавить документ в индекс
  ask <вопрос>      — задать вопрос (можно без слова 'ask')
  list              — список загруженных документов
  clear             — очистить базу
  help              — показать эту справку
  quit / exit       — выход
"""


def main():
    print("=" * 70)
    print("  03_rag_chroma.py — RAG с векторной БД ChromaDB")
    print("=" * 70)
    print(f"  Папка БД:    {CHROMA_DIR}")
    print(f"  Коллекция:   {COLLECTION}")
    print(f"  Записей в БД: {collection.count()}")
    print(HELP)

    while True:
        try:
            line = input("➤ ").strip()
            if not line:
                continue

            # shlex корректно разбирает строки с пробелами и кавычками
            parts = shlex.split(line)
            cmd = parts[0].lower()

            if cmd in ("quit", "exit"):
                break
            elif cmd == "help":
                print(HELP)
            elif cmd == "list":
                cmd_list()
            elif cmd == "clear":
                cmd_clear()
            elif cmd == "index":
                if len(parts) < 2:
                    print("  Использование: index <путь_к_файлу>")
                else:
                    cmd_index(parts[1])
            elif cmd == "ask":
                question = " ".join(parts[1:])
                if not question:
                    print("  Использование: ask <вопрос>")
                else:
                    cmd_ask(question)
            else:
                # По умолчанию — вся строка считается вопросом
                cmd_ask(line)

        except KeyboardInterrupt:
            print("\nВыход.")
            break
        except Exception as e:
            print(f"  ❌ Ошибка: {e}")


if __name__ == "__main__":
    main()
