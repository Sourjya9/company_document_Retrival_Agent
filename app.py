#!/usr/bin/env python3
"""
app.py - Web interface for the CRAG bot
=====================================================================
A simple browser-based chat page. Employees open a URL in their browser
instead of using the terminal.

Setup:
  pip install flask pyopenssl
Run:
  python app.py
Then open https://localhost:5000 in a browser (you'll see a one-time
self-signed certificate warning - click through it, this is expected).
Works on any device on the same network if you replace localhost with
your laptop's local IP, e.g. https://192.168.1.42:5000.
"""

import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template_string
from crag_core import load_index_and_corpus, run_crag

load_dotenv()
APP_PASSWORD = os.getenv("APP_PASSWORD", "changeme123")

app = Flask(__name__)

print("[*] Loading pre-built document index...")
faiss_index, KNOWLEDGE_BASE, SOURCES = load_index_and_corpus()
print(f"[*] Loaded {len(KNOWLEDGE_BASE)} chunks from your company documents.")
print("[*] Web server ready.")

PAGE_HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Company Assistant</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: -apple-system, Segoe UI, Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 16px; background: #f7f7f5; }
    h1 { font-size: 20px; color: #2c2c2a; }
    #chat { background: white; border-radius: 12px; padding: 20px; min-height: 300px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
    .msg { margin-bottom: 16px; line-height: 1.5; }
    .you { color: #185fa5; font-weight: 600; }
    .bot { color: #2c2c2a; }
    .meta { font-size: 12px; color: #888; margin-top: 4px; }
    #inputRow { display: flex; gap: 8px; margin-top: 16px; }
    #question { flex: 1; padding: 10px; border-radius: 8px; border: 1px solid #ccc; font-size: 14px; }
    #askBtn { padding: 10px 20px; border-radius: 8px; border: none; background: #185fa5; color: white; font-size: 14px; cursor: pointer; }
    #askBtn:disabled { background: #aaa; cursor: not-allowed; }
    .loading { color: #888; font-style: italic; }
    input[type=text], input[type=password] { box-sizing: border-box; }
  </style>
</head>
<body>
  <h1>Company Assistant</h1>
  <input type="text" id="username" placeholder="Your name" style="width:100%; padding:10px; border-radius:8px; border:1px solid #ccc; margin-bottom:8px;">
  <input type="password" id="password" placeholder="Enter access password" style="width:100%; padding:10px; border-radius:8px; border:1px solid #ccc; margin-bottom:12px;">
  <div id="chat"></div>
  <div id="inputRow">
    <input type="text" id="question" placeholder="Ask about PTO, benefits, IT policy..." onkeydown="if(event.key==='Enter') ask()">
    <button id="askBtn" onclick="ask()">Ask</button>
  </div>

  <script>
    async function ask() {
      const input = document.getElementById('question');
      const question = input.value.trim();
      if (!question) return;

      const chat = document.getElementById('chat');
      const btn = document.getElementById('askBtn');

      chat.innerHTML += `<div class="msg"><span class="you">You:</span> ${escapeHtml(question)}</div>`;
      const loadingId = 'loading-' + Date.now();
      chat.innerHTML += `<div class="msg loading" id="${loadingId}">Thinking...</div>`;
      chat.scrollTop = chat.scrollHeight;

      input.value = '';
      input.disabled = true;
      btn.disabled = true;

      try {
        const password = document.getElementById('password').value;
        const username = document.getElementById('username').value.trim() || 'Anonymous';

        const res = await fetch('/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question, password, username })
        });
        const data = await res.json();
        document.getElementById(loadingId).remove();

        if (data.error) {
          chat.innerHTML += `<div class="msg bot">Error: ${escapeHtml(data.error)}</div>`;
        } else {
          chat.innerHTML += `
            <div class="msg">
              <span class="bot">${escapeHtml(data.answer)}</span>
              <div class="meta">confidence: ${data.top_confidence.toFixed(2)} &middot; route: ${data.route} &middot; sources: ${data.sources.join(', ')}</div>
            </div>`;
        }
      } catch (e) {
        document.getElementById(loadingId).remove();
        chat.innerHTML += `<div class="msg bot">Error: could not reach the server.</div>`;
      }

      chat.scrollTop = chat.scrollHeight;
      input.disabled = false;
      btn.disabled = false;
      input.focus();
    }

    function escapeHtml(str) {
      const div = document.createElement('div');
      div.textContent = str;
      return div.innerHTML;
    }
  </script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(PAGE_HTML)


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    password = (data.get("password") or "").strip()
    if password != APP_PASSWORD:
        return jsonify({"error": "Incorrect password."}), 401

    question = (data.get("question") or "").strip()
    username = (data.get("username") or "Anonymous").strip()
    if not question:
        return jsonify({"error": "No question provided."}), 400

    result = run_crag(faiss_index, KNOWLEDGE_BASE, SOURCES, question, verbose=False, username=username)
    return jsonify(result)


if __name__ == "__main__":
    # host="0.0.0.0" makes it reachable from other devices on your network,
    # not just this laptop. Remove if you only want to use it yourself.
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, ssl_context="adhoc")