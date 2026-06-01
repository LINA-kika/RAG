"""
02_naive_rag.py — RAG без внешних библиотек (только numpy)
"""

import os
import numpy as np
from openai import OpenAI
import re

# ===================== КОНФИГУРАЦИЯ =====================

LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
EMBEDDING_MODEL = "bge-m3"
LLM_MODEL = "qwen/qwen3-1.7b"
DOC_PATH = os.path.join(os.path.dirname(__file__), "sample_doc.txt")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3
MAX_TOKENS = 512
TEMPERATURE = 0.3

# ===================== ИНИЦИАЛИЗАЦИЯ =====================

client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")

# ===================== ЧАНКИНГ =====================

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    
    return chunks

# ===================== ЭМБЕДДИНГИ =====================

def embed_text(text: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)

def build_index(chunks: list[str]) -> np.ndarray:
    vectors = []
    for chunk in chunks:
        vectors.append(embed_text(chunk))
    return np.vstack(vectors)

# ===================== ПОИСК =====================

def search(query: str, index: np.ndarray, chunks: list[str], k: int = TOP_K):
    q_vec = embed_text(query)
    q_norm = q_vec / np.linalg.norm(q_vec)
    idx_norms = index / np.linalg.norm(index, axis=1, keepdims=True)
    
    similarities = idx_norms @ q_norm
    top_idx = np.argsort(similarities)[::-1][:k]
    
    return [(int(i), float(similarities[i]), chunks[i]) for i in top_idx]

# ===================== БАЙЕСОВСКИЙ СЮРПРИЗ =====================

def bayesian_surprise(query: str, results: list, index: np.ndarray, chunks: list[str]) -> float:
    q_vec = embed_text(query)
    q_norm = q_vec / np.linalg.norm(q_vec)
    idx_norms = index / np.linalg.norm(index, axis=1, keepdims=True)
    
    all_similarities = idx_norms @ q_norm
    avg_similarity = float(np.mean(all_similarities))
    top_similarity = results[0][1]
    
    if avg_similarity > 0:
        surprise_score = (top_similarity - avg_similarity) / avg_similarity
    else:
        surprise_score = 0.0
    
    return surprise_score

# ===================== ДИАЛОГОВЫЙ СЛЕДОВАТЕЛЬ =====================

def detect_uncertainty(answer: str) -> tuple[bool, list[str]]:
    markers = [
        "зависит от", "в некоторых случаях", "обычно", "как правило",
        "может быть", "необходимо уточнить", "возможно", "вероятно"
    ]
    
    found_markers = [m for m in markers if m in answer.lower()]
    needs_clarification = len(found_markers) >= 2 or "зависит от" in answer.lower()
    
    return needs_clarification, found_markers

def interactive_followup(question: str, answer: str, index: np.ndarray, chunks: list[str]) -> str:
    print(f"\nОтвет модели: {answer}")
    print("\nВ ответе есть неопределённость. Задайте уточнение:")
    user_clarification = input("Уточнение: ").strip()
    
    if not user_clarification:
        print("Возвращаю исходный ответ.")
        return answer
    
    enhanced_question = f"{question} (Уточнение: {user_clarification})"
    new_results = search(enhanced_question, index, chunks, k=TOP_K)
    
    context_parts = []
    for idx, sim, text in new_results:
        context_parts.append(f"[Фрагмент #{idx}]\n{text}")
    
    context = "\n\n---\n\n".join(context_parts)
    prompt = PROMPT_TEMPLATE.format(context=context, question=enhanced_question)
    
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    
    new_answer = resp.choices[0].message.content or "[пустой ответ]"
    print(f"\nУточнённый ответ: {new_answer}")
    
    return new_answer

# ===================== ГЕНЕРАЦИЯ =====================

PROMPT_TEMPLATE = """Ты — ассистент, отвечающий на вопросы на основе предоставленного контекста.

КОНТЕКСТ:
{context}

ВОПРОС: {question}

ПРАВИЛА ОТВЕТА:
1. Если в контексте есть точный ответ — дай его.
2. Если точного ответа нет, но есть похожая или связанная информация — дай её и поясни, что это не точный ответ.
3. Если в контексте есть условия (зависит от X, Y) — перечисли их.
4. Только если в контексте вообще нет ничего по теме вопроса — скажи «В документах информации нет».

ОТВЕТ:"""

def ask(question: str, index: np.ndarray, chunks: list[str]) -> str:
    results = search(question, index, chunks, k=TOP_K)
    
    surprise = bayesian_surprise(question, results, index, chunks)
    if surprise > 0.5:
        print(f"Неожиданный запрос! (сюрприз: {surprise:.2f})")
    
    context_parts = []
    for idx, sim, text in results:
        context_parts.append(f"[Фрагмент #{idx}]\n{text}")
    context = "\n\n---\n\n".join(context_parts)
    
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    answer = resp.choices[0].message.content or "[пустой ответ]"
    
    needs_followup, markers = detect_uncertainty(answer)
    if needs_followup:
        answer = interactive_followup(question, answer, index, chunks)
    
    return answer

# ===================== MAIN =====================

def main():
    print("RAG система с коэффициентом уверенности и поиском неточностей")
    print("-" * 50)
    
    with open(DOC_PATH, encoding="utf-8") as f:
        doc = f.read()
    
    chunks = chunk_text(doc)
    print(f"Chunks: {len(chunks)}")
    
    index = build_index(chunks)
    print(f"Index shape: {index.shape}")
    print("-" * 50)
    
    while True:
        try:
            question = input("\nQuestion: ").strip()
            if not question:
                break
            
            answer = ask(question, index, chunks)
            print(f"\nAnswer: {answer}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()