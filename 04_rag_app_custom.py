"""
04_rag_app.py — RAG как веб-приложение (с доработками)

Добавленные функции:
- Анти-галлюцинация (проверка, не выдумала ли модель ответ)
- MMR поиск (разнообразные чанки вместо повторяющихся)
- Кэширование эмбеддингов (ускорение повторных запросов)

Запуск: python 04_rag_app.py
"""

import os
import json
import time
import hashlib
from pathlib import Path
from collections import deque
from typing import List, Dict
import numpy as np
import chromadb
from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI

# ========== КОНФИГУРАЦИЯ ==========

LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"
LLM_MODEL = "google/gemma-4-e4b"

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_app_db")
COLLECTION_NAME = "rag_app"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3
MMR_DIVERSITY = 0.3
MAX_TOKENS = 600
TEMPERATURE = 0.3
PORT = 5011

# ========== ИНИЦИАЛИЗАЦИЯ ==========

app = Flask(__name__)
client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

try:
    chroma_client.delete_collection(COLLECTION_NAME)
except:
    pass

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

# Кэш для эмбеддингов
embedding_cache = {}
cache_file = Path(__file__).parent / "embedding_cache.json"
if cache_file.exists():
    try:
        with open(cache_file, 'r') as f:
            embedding_cache = json.load(f)
        print(f"Загружено {len(embedding_cache)} кэшированных эмбеддингов")
    except:
        pass

# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========

def chunk_text(text: str) -> List[str]:
    """Простое разбиение на чанки с перекрытием (без изменений)"""
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def embed_text(text: str) -> List[float]:
    """Эмбеддинг с кэшированием (НОВАЯ ФУНКЦИЯ)"""
    cache_key = hashlib.md5(text.encode('utf-8')).hexdigest()
    
    if cache_key in embedding_cache:
        return embedding_cache[cache_key]
    
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    embedding = resp.data[0].embedding
    embedding_cache[cache_key] = embedding
    
    # Сохраняем кэш каждые 50 записей
    if len(embedding_cache) % 50 == 0:
        with open(cache_file, 'w') as f:
            json.dump(embedding_cache, f)
    
    return embedding

def embed_batch(texts: List[str]) -> List[List[float]]:
    """Батчевый эмбеддинг"""
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Косинусное сходство"""
    a_np = np.array(a)
    b_np = np.array(b)
    return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))

def mmr_search(query_vec: List[float], k: int = TOP_K, diversity: float = MMR_DIVERSITY) -> List[Dict]:
    """
    MMR поиск — выбирает разнообразные чанки (НОВАЯ ФУНКЦИЯ)
    """
    # Получаем больше кандидатов
    candidates = collection.query(
        query_embeddings=[query_vec],
        n_results=k * 3,
        include=["documents", "metadatas", "distances", "embeddings"]
    )
    
    if not candidates['documents'][0]:
        return []
    
    docs = candidates['documents'][0]
    metas = candidates['metadatas'][0]
    distances = candidates['distances'][0]
    embeddings = candidates['embeddings'][0]
    
    relevance = [1.0 - d for d in distances]
    
    selected = []
    remaining = list(range(len(docs)))
    
    for _ in range(min(k, len(docs))):
        if not remaining:
            break
            
        mmr_scores = []
        for i in remaining:
            rel = relevance[i]
            if selected:
                sim_to_selected = max(cosine_similarity(embeddings[i], embeddings[s]) for s in selected)
            else:
                sim_to_selected = 0
            mmr = diversity * sim_to_selected - (1 - diversity) * rel
            mmr_scores.append(mmr)
        
        best = remaining[np.argmin(mmr_scores)]
        selected.append(best)
        remaining.remove(best)
    
    results = []
    for idx in selected:
        results.append({
            'text': docs[idx],
            'metadata': metas[idx],
            'similarity': relevance[idx]
        })
    
    return results

def confidence_check(answer: str, sources: List[Dict]) -> Dict:
    """
    Проверка на галлюцинации — считает, сколько предложений из ответа
    подтверждаются источниками (НОВАЯ ФУНКЦИЯ)
    """
    sentences = [s.strip() for s in answer.replace('!', '.').replace('?', '.').split('.') if len(s.strip()) > 10]
    
    if not sentences:
        return {'confidence': 1.0, 'is_reliable': True}
    
    verified = 0
    for sent in sentences:
        sent_lower = sent.lower()
        for source in sources:
            source_text = source.get('text', '').lower()
            sent_words = set(sent_lower.split()[:5])
            source_words = set(source_text.split())
            if len(sent_words & source_words) > 0:
                verified += 1
                break
    
    confidence = verified / len(sentences)
    
    return {
        'confidence': round(confidence, 2),
        'is_reliable': confidence > 0.7
    }

# ========== МАРШРУТЫ API ==========

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "Нет файла"}), 400
    
    f = request.files["file"]
    if not f.filename.endswith((".txt", ".md")):
        return jsonify({"error": "Только .txt и .md"}), 400
    
    try:
        text = f.read().decode("utf-8")
    except UnicodeDecodeError:
        return jsonify({"error": "Ошибка кодировки"}), 400
    
    source = f.filename
    chunks = chunk_text(text)
    
    if not chunks:
        return jsonify({"error": "Файл пуст"}), 400
    
    embeddings = embed_batch(chunks)
    
    # Удаляем старые чанки этого файла
    existing = collection.get(where={"source": source})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
    
    ids = [f"{source}::{i}" for i in range(len(chunks))]
    metadatas = [{"source": source, "chunk_index": i} for i, c in enumerate(chunks)]
    
    collection.add(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings)
    
    return jsonify({"source": source, "chunks": len(chunks), "total": collection.count()})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json() or {}
    question = data.get("question", "").strip()
    
    if not question:
        return jsonify({"error": "Пустой вопрос"}), 400
    
    if collection.count() == 0:
        return jsonify({"error": "Нет документов"}), 400
    
    start_time = time.time()
    
    # Получаем эмбеддинг вопроса
    q_vec = embed_text(question)
    
    # MMR поиск (вместо обычного)
    sources = mmr_search(q_vec, k=TOP_K)
    
    if not sources:
        return jsonify({"error": "Ничего не найдено"}), 400
    
    # Собираем контекст
    context_parts = []
    sources_for_ui = []
    for i, src in enumerate(sources, 1):
        context_parts.append(f"[{src['metadata']['source']}, чанк {src['metadata']['chunk_index']}]\n{src['text']}")
        sources_for_ui.append({
            "rank": i,
            "source": src['metadata']['source'],
            "chunk_index": src['metadata']['chunk_index'],
            "similarity": round(src['similarity'], 3),
            "text": src['text'][:300]
        })
    
    context = "\n\n---\n\n".join(context_parts)
    
    prompt = f"""Ты — ассистент. Отвечай строго на основе контекста.
Если ответа нет — скажи «В документах ответа нет».

КОНТЕКСТ:
{context}

ВОПРОС: {question}

ОТВЕТ:"""
    
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    
    answer = resp.choices[0].message.content or "[пустой ответ]"
    
    # Проверка на галлюцинации
    confidence = confidence_check(answer, sources_for_ui)
    
    # Добавляем предупреждение при низкой уверенности
    if not confidence['is_reliable']:
        answer += f"\n\n[Предупреждение: уверенность ответа {confidence['confidence']*100:.0f}%]"
    
    response_time = (time.time() - start_time) * 1000
    
    return jsonify({
        "answer": answer,
        "sources": sources_for_ui,
        "confidence": confidence['confidence'],
        "response_time_ms": round(response_time, 2)
    })

@app.route("/sources", methods=["GET"])
def sources():
    n = collection.count()
    if n == 0:
        return jsonify({"sources": [], "total": 0})
    
    sample = collection.peek(limit=n)
    counts = {}
    for meta in sample["metadatas"]:
        counts[meta["source"]] = counts.get(meta["source"], 0) + 1
    
    return jsonify({"sources": [{"name": k, "chunks": v} for k, v in counts.items()], "total": n})

@app.route("/clear", methods=["POST"])
def clear():
    global collection
    chroma_client.delete_collection(COLLECTION_NAME)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    return jsonify({"status": "ok"})

# ========== HTML ==========

HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>RAG App</title>
<style>
  body { font-family: system-ui; margin: 0; padding: 20px; background: #f5f5f5; }
  .container { max-width: 1000px; margin: 0 auto; }
  .panel { background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  .dropzone { border: 2px dashed #ccc; border-radius: 8px; padding: 30px; text-align: center; cursor: pointer; }
  .dropzone:hover { border-color: #3b82f6; background: #f0f9ff; }
  #chat { height: 400px; overflow-y: auto; border: 1px solid #e5e5e5; border-radius: 8px; padding: 15px; margin-bottom: 15px; }
  .msg { margin-bottom: 15px; padding: 10px 15px; border-radius: 10px; }
  .user { background: #3b82f6; color: white; text-align: right; }
  .bot { background: #e5e5e5; color: black; }
  .ask-row { display: flex; gap: 10px; }
  .ask-row input { flex: 1; padding: 10px; border: 1px solid #ccc; border-radius: 8px; }
  button { padding: 10px 20px; background: #3b82f6; color: white; border: none; border-radius: 8px; cursor: pointer; }
  button:hover { background: #2563eb; }
  .sources { margin-top: 10px; font-size: 12px; color: #666; }
  .sources-toggle { cursor: pointer; color: #8b5cf6; }
  .sources-list { display: none; margin-top: 10px; }
  .sources-list.open { display: block; }
  .source-item { background: #faf5ff; padding: 8px; margin: 5px 0; border-radius: 5px; font-size: 12px; }
  .sidebar { float: right; width: 250px; margin-left: 20px; }
  .main { overflow: hidden; }
  .confidence-low { border-left: 3px solid orange; margin-top: 8px; padding-left: 8px; }
</style>
</head>
<body>
<div class="container">
  <div class="sidebar">
    <div class="panel">
      <h3>Загруженные файлы</h3>
      <div id="sources-list"></div>
      <button id="clear-btn" style="width:100%; margin-top:10px; background:#ef4444;">Очистить всё</button>
    </div>
  </div>
  <div class="main">
    <div class="panel">
      <h2>Загрузить документ</h2>
      <div class="dropzone" id="dropzone">
        Перетащите файл сюда или нажмите для выбора<br>
        <small>(.txt или .md)</small>
        <input type="file" id="file-input" accept=".txt,.md" hidden>
      </div>
    </div>
    <div class="panel">
      <h2>Чат</h2>
      <div id="chat"></div>
      <div class="ask-row">
        <input type="text" id="question" placeholder="Задайте вопрос...">
        <button id="ask-btn">Отправить</button>
      </div>
    </div>
  </div>
</div>

<script>
let mmrDiversity = 0.3;

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const chat = document.getElementById('chat');
const question = document.getElementById('question');
const askBtn = document.getElementById('ask-btn');
const srcList = document.getElementById('sources-list');

function addMsg(text, cls) {
  const div = document.createElement('div');
  div.className = 'msg ' + cls;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

async function uploadFile(file) {
  const form = new FormData();
  form.append('file', file);
  const note = addMsg(`Индексирую "${file.name}"...`, 'bot');
  try {
    const r = await fetch('/upload', { method: 'POST', body: form });
    const d = await r.json();
    if (!r.ok || d.error) {
      note.textContent = 'Ошибка: ' + (d.error || r.status);
      return;
    }
    note.textContent = `Файл "${d.source}" загружен (${d.chunks} чанков)`;
    refreshSources();
  } catch(e) {
    note.textContent = 'Ошибка: ' + e.message;
  }
}

async function refreshSources() {
  try {
    const r = await fetch('/sources');
    const d = await r.json();
    if (!d.sources.length) {
      srcList.innerHTML = '<div>Нет документов</div>';
      return;
    }
    srcList.innerHTML = d.sources.map(s => `<div><b>${s.name}</b><br><small>${s.chunks} чанков</small></div>`).join('');
  } catch(e) {}
}

async function sendQuestion() {
  const q = question.value.trim();
  if (!q) return;
  addMsg(q, 'user');
  question.value = '';
  askBtn.disabled = true;
  const thinking = addMsg('Думаю...', 'bot');
  try {
    const r = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, mmr_diversity: mmrDiversity })
    });
    const d = await r.json();
    if (!r.ok || d.error) {
      thinking.textContent = 'Ошибка: ' + (d.error || r.status);
      return;
    }
    thinking.innerHTML = d.answer.replace(/\n/g, '<br>');
    const metrics = document.createElement('div');
    metrics.style.cssText = 'font-size:11px; color:#666; margin-top:8px;';
    metrics.innerHTML = `Время: ${d.response_time_ms}мс | Уверенность: ${Math.round(d.confidence*100)}%`;
    thinking.appendChild(metrics);
    if (d.sources && d.sources.length) {
      const wrap = document.createElement('div');
      wrap.className = 'sources';
      const toggle = document.createElement('div');
      toggle.className = 'sources-toggle';
      toggle.textContent = '▶ Показать источники';
      const list = document.createElement('div');
      list.className = 'sources-list';
      d.sources.forEach(s => {
        const item = document.createElement('div');
        item.className = 'source-item';
        item.innerHTML = `<b>${s.source}</b> (чанк ${s.chunk_index}, cos=${s.similarity})<br>${s.text}...`;
        list.appendChild(item);
      });
      toggle.onclick = () => {
        list.classList.toggle('open');
        toggle.textContent = list.classList.contains('open') ? '▼ Скрыть источники' : '▶ Показать источники';
      };
      wrap.appendChild(toggle);
      wrap.appendChild(list);
      thinking.appendChild(wrap);
    }
    refreshSources();
  } catch(e) {
    thinking.textContent = 'Ошибка: ' + e.message;
  } finally {
    askBtn.disabled = false;
    question.focus();
  }
}

dropzone.onclick = () => fileInput.click();
dropzone.ondragover = (e) => { e.preventDefault(); dropzone.style.borderColor = '#3b82f6'; };
dropzone.ondragleave = () => { dropzone.style.borderColor = '#ccc'; };
dropzone.ondrop = (e) => {
  e.preventDefault();
  dropzone.style.borderColor = '#ccc';
  if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]);
};
fileInput.onchange = (e) => { if (e.target.files[0]) uploadFile(e.target.files[0]); };
askBtn.onclick = sendQuestion;
question.onkeydown = (e) => { if (e.key === 'Enter') sendQuestion(); };
document.getElementById('clear-btn').onclick = async () => {
  if (confirm('Очистить всё?')) {
    await fetch('/clear', { method: 'POST' });
    refreshSources();
    addMsg('База очищена', 'bot');
  }
};

refreshSources();
addMsg('Добро пожаловать! Загрузите документ и задайте вопрос.', 'bot');
</script>
</body>
</html>"""

if __name__ == "__main__":
    print("RAG App запущен на http://localhost:{}".format(PORT))
    app.run(debug=True, port=PORT)