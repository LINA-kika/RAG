"""
04_rag_app.py — RAG как веб-приложение

RAG-пайплайн с веб-интерфейсом.

Функциональность:
  - Drag-and-drop загрузка .txt файлов
  - Чанкинг и индексирование
  - Чат-интерфейс с историей
  - Прозрачность: отображение использованных фрагментов
  - Список загруженных источников
  - Байесовский сюрприз для детекции неожиданных запросов
  - Интерактивное уточнение при неопределённых ответах

Endpoints:
  GET  /            - HTML фронтенд
  POST /upload      - загрузка файла
  POST /ask         - вопрос {question} -> {answer, sources, surprise}
  GET  /sources     - список документов
  POST /clear       - очистка базы
"""

import os
import chromadb
from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI
import numpy as np


# ===================== КОНФИГУРАЦИЯ =====================

LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
EMBEDDING_MODEL = "bge-m3"
LLM_MODEL = "qwen/qwen3-1.7b"

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_app_db")
COLLECTION = "rag_app"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3

MAX_TOKENS = 600
TEMPERATURE = 0.3
SURPRISE_THRESHOLD = 0.5
PORT = 5011


# ===================== ИНИЦИАЛИЗАЦИЯ =====================

app = Flask(__name__)
client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
chroma = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma.get_or_create_collection(
    name=COLLECTION, metadata={"hnsw:space": "cosine"}
)


# ===================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =====================

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


def embed_batch(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def embed_one(text: str) -> list[float]:
    return embed_batch([text])[0]


def get_all_embeddings() -> np.ndarray:
    if collection.count() == 0:
        return np.array([])
    all_data = collection.get(include=["embeddings"])
    if all_data["embeddings"] is None or len(all_data["embeddings"]) == 0:
        return np.array([])
    return np.array(all_data["embeddings"], dtype=np.float32)


def bayesian_surprise(query: str, top_similarity: float) -> float:
    if collection.count() == 0:
        return 0.0
    
    all_embeddings = get_all_embeddings()
    if len(all_embeddings) == 0:
        return 0.0
    
    try:
        q_vec = np.array(embed_one(query), dtype=np.float32)
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-8)
        idx_norms = all_embeddings / (np.linalg.norm(all_embeddings, axis=1, keepdims=True) + 1e-8)
        all_similarities = idx_norms @ q_norm
        avg_similarity = float(np.mean(all_similarities))
        
        if avg_similarity > 0:
            return (top_similarity - avg_similarity) / avg_similarity
        return 0.0
    except Exception as e:
        print(f"Error in bayesian_surprise: {e}")
        return 0.0


def detect_uncertainty(answer: str) -> tuple[bool, list[str]]:
    markers = [
        "зависит от", "в некоторых случаях", "обычно", "как правило",
        "может быть", "необходимо уточнить", "возможно", "вероятно",
        "скорее всего", "предположительно"
    ]
    found_markers = [m for m in markers if m in answer.lower()]
    needs_clarification = len(found_markers) >= 2 or "зависит от" in answer.lower()
    return needs_clarification, found_markers


# ===================== МАРШРУТЫ API =====================

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    f = request.files["file"]
    if not f.filename.lower().endswith((".txt", ".md")):
        return jsonify({"error": "Only .txt and .md files are supported"}), 400

    try:
        text = f.read().decode("utf-8")
    except UnicodeDecodeError:
        return jsonify({"error": "File must be UTF-8 encoded"}), 400

    source = f.filename
    chunks = chunk_text(text)
    
    if not chunks:
        return jsonify({"error": "File is empty"}), 400

    embeddings = embed_batch(chunks)

    existing = collection.get(where={"source": source})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    ids = [f"{source}::{i}" for i in range(len(chunks))]
    metadatas = [{"source": source, "chunk_index": i, "length": len(c)} for i, c in enumerate(chunks)]

    collection.add(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings)

    return jsonify({
        "source": source,
        "chunks": len(chunks),
        "total_in_db": collection.count(),
    })


# ===================== ПРОМПТЫ =====================

PROMPT_TEMPLATE = """Ты - ассистент, который отвечает на вопросы на основе предоставленного контекста.

КОНТЕКСТ:
{context}

ВОПРОС: {question}

ПРАВИЛА:
1. Найди в контексте информацию, которая отвечает на вопрос.
2. Дай ответ, используя факты из контекста.
3. Если в контексте есть число или конкретное значение - обязательно укажи его.
4. Отвечай кратко, по делу.

ОТВЕТ:"""


FOLLOWUP_PROMPT_TEMPLATE = """Уточни ответ на вопрос, используя контекст ниже.

КОНТЕКСТ:
{context}

ВОПРОС: {question}

КОНКРЕТНЫЙ ОТВЕТ:"""


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json() or {}
    question = (data.get("question") or "").strip()
    is_followup = data.get("followup", False)
    original_question = data.get("original_question", "")
    
    if not question:
        return jsonify({"error": "Empty question"}), 400

    if collection.count() == 0:
        return jsonify({"error": "Database is empty. Please upload a document first."}), 400

    q_vec = embed_one(question)
    results = collection.query(query_embeddings=[q_vec], n_results=TOP_K)
    
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    context_parts = []
    sources_for_ui = []
    top_similarity = 0.0
    
    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
        similarity = 1.0 - dist
        if rank == 1:
            top_similarity = similarity
        context_parts.append(doc)
        sources_for_ui.append({
            "rank": rank,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "similarity": round(similarity, 3),
            "text": doc,
        })
    
    context = "\n\n---\n\n".join(context_parts)
    surprise = bayesian_surprise(question, top_similarity)

    if is_followup and original_question:
        prompt_template = FOLLOWUP_PROMPT_TEMPLATE
        display_question = f"{original_question} (Уточнение: {question})"
    else:
        prompt_template = PROMPT_TEMPLATE
        display_question = question

    prompt = prompt_template.format(context=context, question=display_question)
    
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        
        msg = resp.choices[0].message
        answer = msg.content or "[пустой ответ]"
    except Exception as e:
        print(f"Error in LLM call: {e}")
        answer = f"Ошибка при вызове модели: {str(e)}"

    needs_followup = False
    uncertainty_markers = []
    
    if not is_followup and answer != "[пустой ответ]":
        needs_followup, uncertainty_markers = detect_uncertainty(answer)

    return jsonify({
        "answer": answer,
        "sources": sources_for_ui,
        "surprise": round(surprise, 3),
        "needs_followup": needs_followup,
        "uncertainty_markers": uncertainty_markers,
        "is_unexpected": surprise > SURPRISE_THRESHOLD,
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

    return jsonify({
        "sources": [{"name": k, "chunks": v} for k, v in counts.items()],
        "total": n,
    })


@app.route("/clear", methods=["POST"])
def clear():
    global collection
    chroma.delete_collection(COLLECTION)
    collection = chroma.get_or_create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    return jsonify({"status": "ok"})


# ===================== HTML ФРОНТЕНД =====================

HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>RAG Application</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  :root {
    --bg: #f1f5f9;
    --card: #fff;
    --border: #e2e8f0;
    --text: #1e293b;
    --muted: #64748b;
    --primary: #3b82f6;
    --purple: #8b5cf6;
    --orange: #f97316;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
  header { background: var(--card); border-bottom: 1px solid var(--border); padding: 14px 24px; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 18px; font-weight: 700; flex: 1; }
  .badge { font-size: 12px; color: var(--muted); padding: 3px 10px; border-radius: 20px; background: #f1f5f9; }
  .layout { max-width: 1100px; margin: 0 auto; padding: 18px 16px; display: grid; grid-template-columns: 1fr 280px; gap: 16px; }
  .panel { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
  .panel h2 { font-size: 13px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: .4px; margin-bottom: 10px; }
  .dropzone { border: 2px dashed var(--border); border-radius: 10px; padding: 22px; text-align: center; cursor: pointer; transition: all .2s; background: #fafafa; }
  .dropzone:hover, .dropzone.over { border-color: var(--primary); background: #eff6ff; }
  .dropzone p { color: var(--muted); font-size: 13px; }
  .dropzone strong { color: var(--text); }
  #chat { display: flex; flex-direction: column; gap: 10px; min-height: 200px; max-height: 460px; overflow-y: auto; padding: 4px; }
  .msg { padding: 10px 14px; border-radius: 10px; line-height: 1.6; font-size: 14px; }
  .msg.user { align-self: flex-end; background: var(--primary); color: #fff; max-width: 78%; }
  .msg.bot { align-self: flex-start; background: #f8fafc; border: 1px solid var(--border); max-width: 88%; white-space: pre-wrap; }
  .msg.error { align-self: center; background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; font-size: 13px; }
  .surprise-badge { display: inline-block; background: var(--orange); color: #fff; font-size: 10px; padding: 2px 8px; border-radius: 12px; margin-bottom: 6px; }
  .uncertainty-badge { display: inline-block; background: var(--purple); color: #fff; font-size: 10px; padding: 2px 8px; border-radius: 12px; margin-bottom: 6px; margin-left: 6px; }
  .followup-btn { background: none; border: 1px solid var(--purple); color: var(--purple); font-size: 11px; padding: 4px 10px; border-radius: 16px; margin-top: 8px; cursor: pointer; transition: all .2s; }
  .followup-btn:hover { background: var(--purple); color: #fff; }
  .sources { margin-top: 8px; border-top: 1px dashed var(--border); padding-top: 8px; }
  .sources-toggle { font-size: 12px; color: var(--purple); cursor: pointer; user-select: none; }
  .sources-toggle:hover { text-decoration: underline; }
  .sources-list { display: none; margin-top: 8px; flex-direction: column; gap: 6px; }
  .sources-list.open { display: flex; }
  .source-item { background: #faf5ff; border: 1px solid #e9d5ff; border-radius: 6px; padding: 8px 10px; font-size: 12px; }
  .source-item .meta { color: var(--purple); font-weight: 700; margin-bottom: 4px; }
  .source-item .text { color: var(--text); line-height: 1.55; max-height: 80px; overflow-y: auto; }
  .ask-row { display: flex; gap: 8px; margin-top: 12px; }
  .ask-row input { flex: 1; border: 1px solid var(--border); border-radius: 8px; padding: 9px 12px; font-size: 14px; font-family: inherit; outline: none; }
  .ask-row input:focus { border-color: var(--primary); }
  .btn { padding: 9px 18px; border: none; border-radius: 8px; font-size: 13px; font-weight: 600; cursor: pointer; transition: opacity .2s; }
  .btn:disabled { opacity: .45; cursor: not-allowed; }
  .btn-primary { background: var(--primary); color: #fff; }
  .btn-clear { background: #fee2e2; color: #b91c1c; }
  .btn:hover:not(:disabled) { opacity: .85; }
  .src-list { display: flex; flex-direction: column; gap: 6px; }
  .src-item { background: #f8fafc; border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px; font-size: 12px; }
  .src-item b { color: var(--text); }
  .src-item span { color: var(--muted); font-size: 11px; }
  .empty { color: var(--muted); font-size: 12px; font-style: italic; padding: 6px; }
  .toast { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); background: #1e293b; color: #fff; padding: 9px 18px; border-radius: 8px; font-size: 13px; opacity: 0; transition: opacity .3s; pointer-events: none; }
  .toast.show { opacity: 1; }
  .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid var(--border); border-top-color: var(--primary); border-radius: 50%; animation: spin .7s linear infinite; vertical-align: middle; margin-right: 6px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); justify-content: center; align-items: center; z-index: 1000; }
  .modal-content { background: var(--card); border-radius: 12px; padding: 20px; max-width: 500px; width: 90%; }
  .modal-content h3 { margin-bottom: 12px; }
  .modal-content input { width: 100%; padding: 10px; margin: 12px 0; border: 1px solid var(--border); border-radius: 8px; }
  .modal-buttons { display: flex; gap: 8px; justify-content: flex-end; }
  @media (max-width: 800px) { .layout { grid-template-columns: 1fr; } }
</style>
</head>
<body>

<header>
  <span style="font-size: 22px;">RAG</span>
  <h1>RAG Application</h1>
  <span class="badge">Local RAG with Bayesian Surprise</span>
</header>

<div class="layout">
  <main style="display: flex; flex-direction: column; gap: 14px;">
    <div class="panel">
      <h2>Upload Document</h2>
      <div class="dropzone" id="dropzone">
        <p><strong>Drag and drop .txt or .md file here</strong><br>or <a href="#" id="pick">select manually</a></p>
        <input type="file" id="file-input" accept=".txt,.md" hidden>
      </div>
    </div>

    <div class="panel">
      <h2>Chat</h2>
      <div id="chat">
        <div class="msg bot">Загрузите документ и задайте вопрос.</div>
      </div>
      <div class="ask-row">
        <input type="text" id="question" placeholder="Задайте вопрос..." />
        <button class="btn btn-primary" id="ask-btn">Ask</button>
      </div>
    </div>
  </main>

  <aside style="display: flex; flex-direction: column; gap: 14px;">
    <div class="panel">
      <h2>Sources</h2>
      <div id="sources-list" class="src-list"></div>
      <button class="btn btn-clear" id="clear-btn" style="margin-top: 12px; width: 100%; font-size: 12px;">Clear Database</button>
    </div>
  </aside>
</div>

<div class="toast" id="toast"></div>

<div id="followup-modal" class="modal">
  <div class="modal-content">
    <h3>Clarification Required</h3>
    <p id="original-question-display" style="color: var(--muted); font-size: 13px;"></p>
    <p>What would you like to clarify?</p>
    <input type="text" id="followup-input" placeholder="Enter clarification..." />
    <div class="modal-buttons">
      <button class="btn" id="cancel-followup" style="background:#e2e8f0;">Cancel</button>
      <button class="btn btn-primary" id="submit-followup">Submit</button>
    </div>
  </div>
</div>

<script>
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const chat = document.getElementById('chat');
const question = document.getElementById('question');
const askBtn = document.getElementById('ask-btn');
const srcList = document.getElementById('sources-list');
const modal = document.getElementById('followup-modal');
const followupInput = document.getElementById('followup-input');
let pendingQuestion = null;

function escHtml(s) {
  return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function addMsg(text, cls) {
  const div = document.createElement('div');
  div.className = 'msg ' + cls;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

dropzone.addEventListener('click', () => fileInput.click());
document.getElementById('pick').addEventListener('click', e => { e.preventDefault(); fileInput.click(); });
fileInput.addEventListener('change', e => { if (e.target.files[0]) uploadFile(e.target.files[0]); });

['dragenter', 'dragover'].forEach(ev => dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.add('over'); }));
['dragleave', 'drop'].forEach(ev => dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.remove('over'); }));
dropzone.addEventListener('drop', e => { if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]); });

async function uploadFile(file) {
  const form = new FormData();
  form.append('file', file);
  const note = addMsg(`Indexing "${file.name}"...`, 'bot');
  try {
    const r = await fetch('/upload', { method: 'POST', body: form });
    const d = await r.json();
    if (!r.ok || d.error) {
      note.className = 'msg error';
      note.textContent = 'Error: ' + (d.error || r.status);
      return;
    }
    note.textContent = `Added "${d.source}" - ${d.chunks} chunks. Total: ${d.total_in_db}.`;
    refreshSources();
  } catch (e) {
    note.className = 'msg error';
    note.textContent = 'Error: ' + e.message;
  }
}

async function refreshSources() {
  try {
    const r = await fetch('/sources');
    const d = await r.json();
    if (!d.sources.length) {
      srcList.innerHTML = '<div class="empty">No documents</div>';
      return;
    }
    srcList.innerHTML = d.sources.map(s => `<div class="src-item"><b>${escHtml(s.name)}</b><br><span>${s.chunks} chunks</span></div>`).join('');
  } catch (e) {}
}

async function sendFollowup() {
  const clarification = followupInput.value.trim();
  if (!clarification) {
    modal.style.display = 'none';
    return;
  }
  modal.style.display = 'none';
  followupInput.value = '';
  addMsg(`Clarification: ${clarification}`, 'user');
  const thinking = addMsg('', 'bot');
  thinking.innerHTML = '<span class="spinner"></span>Thinking...';
  try {
    const r = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: clarification, followup: true, original_question: pendingQuestion })
    });
    const d = await r.json();
    if (!r.ok || d.error) {
      thinking.className = 'msg error';
      thinking.textContent = 'Error: ' + (d.error || r.status);
      return;
    }
    thinking.innerHTML = '';
    const ansDiv = document.createElement('div');
    ansDiv.textContent = d.answer;
    thinking.appendChild(ansDiv);
    if (d.sources && d.sources.length) {
      const wrap = document.createElement('div');
      wrap.className = 'sources';
      const toggle = document.createElement('div');
      toggle.className = 'sources-toggle';
      toggle.textContent = `Show sources (${d.sources.length})`;
      const list = document.createElement('div');
      list.className = 'sources-list';
      d.sources.forEach(s => {
        const item = document.createElement('div');
        item.className = 'source-item';
        item.innerHTML = `<div class="meta">#${s.rank} | ${escHtml(s.source)} | chunk ${s.chunk_index} | cos=${s.similarity}</div><div class="text">${escHtml(s.text)}</div>`;
        list.appendChild(item);
      });
      toggle.addEventListener('click', () => {
        const open = list.classList.toggle('open');
        toggle.textContent = open ? `Hide sources (${d.sources.length})` : `Show sources (${d.sources.length})`;
      });
      wrap.appendChild(toggle);
      wrap.appendChild(list);
      thinking.appendChild(wrap);
    }
  } catch (e) {
    thinking.className = 'msg error';
    thinking.textContent = 'Error: ' + e.message;
  }
}

async function sendQuestion() {
  const q = question.value.trim();
  if (!q) return;
  addMsg(q, 'user');
  pendingQuestion = q;
  question.value = '';
  askBtn.disabled = true;
  const thinking = addMsg('', 'bot');
  thinking.innerHTML = '<span class="spinner"></span>Thinking...';
  try {
    const r = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, followup: false })
    });
    const d = await r.json();
    if (!r.ok || d.error) {
      thinking.className = 'msg error';
      thinking.textContent = 'Error: ' + (d.error || r.status);
      return;
    }
    thinking.innerHTML = '';
    const badgeContainer = document.createElement('div');
    if (d.is_unexpected) {
      const surpriseBadge = document.createElement('span');
      surpriseBadge.className = 'surprise-badge';
      surpriseBadge.textContent = `Unexpected query! Surprise: ${d.surprise}`;
      badgeContainer.appendChild(surpriseBadge);
    }
    if (d.needs_followup) {
      const uncertaintyBadge = document.createElement('span');
      uncertaintyBadge.className = 'uncertainty-badge';
      uncertaintyBadge.textContent = `Clarification needed`;
      badgeContainer.appendChild(uncertaintyBadge);
    }
    thinking.appendChild(badgeContainer);
    const ansDiv = document.createElement('div');
    ansDiv.textContent = d.answer;
    thinking.appendChild(ansDiv);
    if (d.needs_followup) {
      const followupBtn = document.createElement('button');
      followupBtn.className = 'followup-btn';
      followupBtn.textContent = 'Clarify answer';
      followupBtn.onclick = () => {
        document.getElementById('original-question-display').innerHTML = `<strong>Original question:</strong> ${escHtml(q)}<br><strong>Answer:</strong> ${escHtml(d.answer.substring(0, 150))}...`;
        followupInput.value = '';
        modal.style.display = 'flex';
      };
      thinking.appendChild(followupBtn);
    }
    if (d.sources && d.sources.length) {
      const wrap = document.createElement('div');
      wrap.className = 'sources';
      const toggle = document.createElement('div');
      toggle.className = 'sources-toggle';
      toggle.textContent = `Show sources (${d.sources.length})`;
      const list = document.createElement('div');
      list.className = 'sources-list';
      d.sources.forEach(s => {
        const item = document.createElement('div');
        item.className = 'source-item';
        item.innerHTML = `<div class="meta">#${s.rank} | ${escHtml(s.source)} | chunk ${s.chunk_index} | cos=${s.similarity}</div><div class="text">${escHtml(s.text)}</div>`;
        list.appendChild(item);
      });
      toggle.addEventListener('click', () => {
        const open = list.classList.toggle('open');
        toggle.textContent = open ? `Hide sources (${d.sources.length})` : `Show sources (${d.sources.length})`;
      });
      wrap.appendChild(toggle);
      wrap.appendChild(list);
      thinking.appendChild(wrap);
    }
  } catch (e) {
    thinking.className = 'msg error';
    thinking.textContent = 'Error: ' + e.message;
  } finally {
    askBtn.disabled = false;
    question.focus();
  }
}

askBtn.addEventListener('click', sendQuestion);
question.addEventListener('keydown', e => { if (e.key === 'Enter') sendQuestion(); });
document.getElementById('clear-btn').addEventListener('click', async () => {
  if (!confirm('Delete all documents?')) return;
  await fetch('/clear', { method: 'POST' });
  refreshSources();
  showToast('Database cleared');
});
document.getElementById('submit-followup').addEventListener('click', sendFollowup);
document.getElementById('cancel-followup').addEventListener('click', () => { modal.style.display = 'none'; });
refreshSources();
</script>
</body>
</html>"""


# ===================== ТОЧКА ВХОДА =====================

if __name__ == "__main__":
    print(f"RAG Application running on http://localhost:{PORT}")
    print(f"Database: {CHROMA_DIR}")
    print(f"Records in database: {collection.count()}")
    app.run(debug=False, port=PORT)