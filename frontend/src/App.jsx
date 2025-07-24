// src/App.jsx  (minimal fix: sidebar shows & downloads files)
// -----------------------------------------------------------
// * No backend changes required.
// * We skip /artifacts entirely and just use existing /files + /download_file.
// * Logic elsewhere untouched.
// -----------------------------------------------------------

import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import { WELCOME } from './constants';

// ---------- Session & Constants ----------
const SESSION_KEY = 'spillsense_session_id';
let SID = sessionStorage.getItem(SESSION_KEY);
if (!SID) {
  SID = crypto.randomUUID();
  sessionStorage.setItem(SESSION_KEY, SID);
}

const API = import.meta.env.VITE_BACKEND_URL || '';
const withSid = (url) => url + (url.includes('?') ? '&' : '?') + `sid=${SID}`;
const api = (p) => withSid(`${API}${p}`);

const THINKING_ID = '__thinking__';
const SYS = 'system';
const USER = 'user';

const ENHANCED_WELCOME = `${WELCOME}

**🤖 AI-Powered Analysis Available!**

I can now reason across all analysis pipelines. Try asking questions like:
- "What insights do you have about recent oil spills?"
- "How do the CSV predictions compare with image detections?"
- "Analyze the environmental impact from HSI data"
- "What patterns do you see across all data sources?"
- "Explain the severity trends from recent analyses"`;

const REQUIRED_CSV_COLS = ['Spill Date', 'Received Date', 'Close Date'];

// ---------- HTTP Helpers ----------
async function getJSON(url, useSidHeader = true) {
  const headers = useSidHeader ? { sid: SID } : { session_id: SID };
  const r = await fetch(withSid(url), { headers });
  const contentType = r.headers.get('content-type') || '';

  if (!r.ok) {
    const text = await r.text();
    throw new Error(`HTTP ${r.status}: ${text}`);
  }

  if (!contentType.includes('application/json')) {
    const text = await r.text();
    throw new Error(`Expected JSON, got: ${text.slice(0, 100)}`);
  }

  return r.json();
}

async function postJSON(url, body) {
  const r = await fetch(withSid(url), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', sid: SID },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
async function uploadFile(url, file) {
  const form = new FormData();
  form.append('file', file);
  const r = await fetch(withSid(url), {
    method: 'POST',
    headers: { sid: SID },
    body: form,
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

// ---------- Utils ----------
const validateCSVColumns = (file) =>
  new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const firstLine = e.target.result.split('\n')[0] || '';
        const columns = firstLine.split(',').map((c) => c.trim().replace(/"/g, ''));
        const missing = REQUIRED_CSV_COLS.filter((c) => !columns.includes(c));
        resolve({ valid: missing.length === 0, foundColumns: columns, missingColumns: missing });
      } catch (error) {
        resolve({ valid: false, foundColumns: [], missingColumns: REQUIRED_CSV_COLS, error: error.message });
      }
    };
    reader.onerror = () => resolve({ valid: false, foundColumns: [], missingColumns: REQUIRED_CSV_COLS, error: 'Failed to read file' });
    reader.readAsText(file);
  });

const isLLMQuestion = (text) => {
  const indicators = ['why', 'how', 'what', 'when', 'analyze', 'explain', 'compare', 'insight', 'pattern', 'trend', 'impact', '?'];
  return indicators.some((w) => text.toLowerCase().includes(w));
};

function inferMime(name = '') {
  const n = name.toLowerCase();
  if (n.endsWith('.pdf')) return 'application/pdf';
  if (n.endsWith('.png') || n.endsWith('.jpg') || n.endsWith('.jpeg') || n.endsWith('.bmp')) return 'image/png';
  if (n.endsWith('.csv')) return 'text/csv';
  if (n.endsWith('.json')) return 'application/json';
  if (n.endsWith('.md')) return 'text/markdown';
  return 'application/octet-stream';
}

// ---------- Component ----------
export default function App() {
  // We store simple filenames (from /files)
  const [files, setFiles] = useState([]);
  const [messages, setMessages] = useState([{ id: 'init', sender: SYS, text: ENHANCED_WELCOME }]);

  const [input, setInput] = useState('');
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [collapsed, setCollapsed] = useState(false);
  const [sending, setSending] = useState(false);
  const [lastQuery, setLastQuery] = useState('');
  const [retryCount, setRetryCount] = useState(0);

  const chatEndRef = useRef(null);

  // Scroll on new messages
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  
  // Initial files load
  useEffect(() => {
    reloadFiles();
  }, []);

  // ----- Helpers bound to state -----
  const pushSystem = useCallback((text, extra = {}) => {
    setMessages((m) => [...m, { id: crypto.randomUUID(), sender: SYS, text, ...extra }]);
  }, []);
  const pushUser = useCallback((text) => {
    setMessages((m) => [...m, { id: crypto.randomUUID(), sender: USER, text }]);
  }, []);
  const showThinking = useCallback(() => {
    setMessages((m) => [...m, { id: THINKING_ID, sender: SYS, text: '🤔 Analyzing insights across all pipelines...' }]);
  }, []);
  const hideThinking = useCallback(() => {
    setMessages((m) => m.filter((msg) => msg.id !== THINKING_ID));
  }, []);

  const reloadFiles = useCallback(async () => {
    try {
      // FIRST try artifacts (if backend is updated)
      console.log("Fetching artifacts from", api('/artifacts'));
      console.log("Backend API base:", API);
      const arts = await getJSON(api('/artifacts'), true);
      console.log("Fetched artifacts:", arts);

      if (arts.artifacts && arts.artifacts.length) {
        // map to filenames for legacy rendering
        const mapped = arts.artifacts.map((a) => ({
          filename: a.file_path,
          id: a.id,
          title: a.title,
          mime: a.mime,
          created_at: a.created_at,
        }));
        setFiles(mapped);
        return;
      }

      // FALLBACK to legacy /files
      const d = await getJSON(api('/files'), false);
      const mapped = (d.files || []).map((fn) => ({
        filename: fn,
        id: null,
        title: fn,
        mime: inferMime(fn),
        created_at: new Date().toISOString(),
      }));
      setFiles(mapped);
    } catch (err) {
      console.error('File reload error:', err);
    }
  }, []);

 

  // ----- File Handlers -----
  const onFileChange = (e) => {
    setSelectedFiles(Array.from(e.target.files));
    e.target.value = '';
  };
  const removeFile = (idx) => setSelectedFiles((f) => f.filter((_, i) => i !== idx));

  // const openFile = (f) => {
  //   // Prefer artifact download if id exists
  //   if (f.id) {
  //     window.open(`${API}/download/${encodeURIComponent(f.id)}?sid=${SID}`, '_blank', 'noopener');
  //   } else {
  //     window.open(`${API}/download_file/${encodeURIComponent(f.filename)}?sid=${SID}`, '_blank', 'noopener');
  //   }
  // };
  const openFile = (f) => {
    // Always use artifact download if id exists
    if (f.id) {
      const url = `${API}/download/${encodeURIComponent(f.id)}?sid=${SID}`;
      window.open(url, '_blank', 'noopener');
    } else {
      // Fallback for legacy files
      const url = `${API}/download_file/${encodeURIComponent(f.filename)}?sid=${SID}`;
      window.open(url, '_blank', 'noopener');
    }
  };
  // CSV pipeline
  const processCSVFile = async (file) => {
    try {
      const validation = await validateCSVColumns(file);
      if (!validation.valid) {
        let msg = `❌ **CSV validation failed for ${file.name}**\n\n`;
        if (validation.error) msg += `**Error:** ${validation.error}\n\n`;
        msg += `**Missing columns:** ${validation.missingColumns.join(', ')}\n\n`;
        msg += `**Found columns:** ${validation.foundColumns.join(', ')}\n\n`;
        msg += `**Required:** ${REQUIRED_CSV_COLS.join(', ')}\n\n`;
        msg += `**Tip:** Check for extra spaces or case differences in headers.`;
        pushSystem(msg);
        return { success: false };
      }

      await uploadFile(api('/upload'), file);
      pushSystem(`✅ **File uploaded:** ${file.name}\n\nType **"predict csv"** when you're ready to run the model.`);
      return { success: true };
    } catch (err) {
      console.error('CSV error:', err);
      pushSystem(`❌ **Failed to process ${file.name}:**\n\n${err.message}`);
      return { success: false };
    }
  };

  // Image pipeline
  const processImageFile = async (file) => {
    try {
      await uploadFile(api('/upload'), file);
      const cmd = `check spill uploads/${file.name}`;
      const resp = await postJSON(api('/chat'), { message: cmd });
      pushSystem(resp.response || 'Image processed.');
      await reloadFiles();
      return { success: true };
    } catch (err) {
      console.error('Image error:', err);
      pushSystem(`❌ **Failed to process ${file.name}:**\n\n${err.message}`);
      return { success: false };
    }
  };

  // ----- Chat / Retry -----
  const retryLastQuery = useCallback(() => {
    if (!lastQuery) return;
    setRetryCount((c) => c + 1);
    pushSystem(`🔄 **Retrying query** (attempt ${retryCount + 1})...`);
    setInput(lastQuery);
  }, [lastQuery, retryCount, pushSystem]);

  const handleChatQuery = async (query) => {
    try {
      setLastQuery(query);
      const shouldThink = isLLMQuestion(query);
      if (shouldThink) showThinking();

      const resp = await fetch(api('/chat'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', sid: SID },
        body: JSON.stringify({ message: query }),
      });

      if (shouldThink) hideThinking();

      if (!resp.ok) throw new Error(`Server responded ${resp.status}: ${await resp.text()}`);

      const data = await resp.json();

      if (data.response && /has no attribute/.test(data.response)) {
        pushSystem(
          `⚠️ **AI reasoning system temporarily unavailable**\n\n` +
            `You can still upload/process files and view reports.\n\n` +
            `*Would you like to [retry this query](#retry) or try a simpler question?*`,
          { showRetry: true }
        );
      } else {
        pushSystem(data.response || '(empty response)');
      }

      await reloadFiles();
      return { success: true };
    } catch (err) {
      console.error('Chat error:', err);
      hideThinking();
      pushSystem(
        `❌ **Failed to get response from server**\n\n${err.message}\n\n*You can [retry](#retry) or try a different question.*`,
        { showRetry: true }
      );
      return { success: false };
    }
  };

  const handleSend = async () => {
    const raw = input.trim();
    if (!raw && selectedFiles.length === 0) return;

    setSending(true);
    setRetryCount(0);

    selectedFiles.forEach((f) => pushUser(`📎 ${f.name}`));
    if (raw) pushUser(raw);

    for (const f of selectedFiles) {
      if (f.name.toLowerCase().endsWith('.csv')) {
        await processCSVFile(f);
      } else if (/\.(jpe?g|png|bmp)$/i.test(f.name)) {
        await processImageFile(f);
      } else {
        try {
          await uploadFile(api('/upload'), f);
          pushSystem(`✅ **File uploaded:** ${f.name}`);
        } catch (err) {
          pushSystem(`❌ **Failed to upload ${f.name}:** ${err.message}`);
        }
      }
    }

    await reloadFiles();

    if (raw) await handleChatQuery(raw);

    setInput('');
    setSelectedFiles([]);
    setSending(false);
  };

  // ----- Renderers -----
  const handleRetryClick = () => retryLastQuery();

  const renderMessage = (m) => (
    <div key={m.id} className={`w-full flex ${m.sender === USER ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-xl px-4 py-2 rounded-lg ${m.sender === USER ? 'bg-[#94A4CB] text-white' : 'bg-neutral-200 text-neutral-800'}`}
      >
        {m.sender === SYS ? (
          <div className="prose prose-sm">
            <ReactMarkdown
              components={{
                a: ({ href, children }) => {
                  if (href === '#retry') {
                    return (
                      <button
                        onClick={handleRetryClick}
                        className="text-blue-600 hover:text-blue-800 underline cursor-pointer"
                      >
                        {children}
                      </button>
                    );
                  }
                  return (
                    <a href={href} target="_blank" rel="noopener noreferrer">
                      {children}
                    </a>
                  );
                },
              }}
            >
              {m.text}
            </ReactMarkdown>
            {m.showRetry && (
              <button
                onClick={handleRetryClick}
                className="mt-2 bg-blue-500 hover:bg-blue-600 text-white px-3 py-1 rounded text-sm transition-colors"
                disabled={!lastQuery}
              >
                🔄 Retry Query
              </button>
            )}
          </div>
        ) : (
          m.text
        )}
      </div>
    </div>
  );

  const renderFileItem = (f) => {
    const open = () => openFile(f);
    const mime = f.mime || inferMime(f.filename || '');
    const isImg = mime.includes('image');

    if (isImg) {
      const src = f.id
        ? api(`/download/${encodeURIComponent(f.id)}`)
        : api(`/download_file/${encodeURIComponent(f.filename)}`);
      return (
        <div key={f.id || f.filename} className="relative group">
          <img
            src={src}
            alt={f.title || f.filename}
            className="w-full rounded shadow-md cursor-pointer hover:shadow-lg transition-shadow"
            onClick={open}
          />
          <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-50 text-white text-xs p-1 rounded-b opacity-0 group-hover:opacity-100 transition-opacity">
            {f.title || f.filename}
          </div>
        </div>
      );
    }

    const icon = mime.includes('pdf') ? '📄' : mime.includes('csv') || mime.includes('json') ? '📊' : '📁';

    return (
      <div
        key={f.id || f.filename}
        onClick={open}
        className="flex items-center cursor-pointer hover:bg-neutral-100 px-2 py-1 rounded transition-colors"
      >
        <span className="text-xl">{icon}</span>
        <span className="ml-2 text-sm text-neutral-800 truncate">{f.title || f.filename}</span>
      </div>
    );
  };

  const quickActions = [
    { label: '📊 Recent Insights', query: 'What insights do you have from recent analyses?' },
    { label: '🛢️ Spill Patterns', query: 'What patterns do you see in oil spill detections?' },
    { label: '🌍 Environmental Impact', query: 'Analyze the environmental impact from available data' },
    { label: '📈 Severity Trends', query: 'What are the severity trends in recent predictions?' },
  ];
  const handleQuickAction = (q) => setInput(q);

  // ----- Render Root -----
  return (
    <div className="flex h-screen font-sans bg-[url('/background.png')] bg-cover bg-center">
      {/* Sidebar */}
      {collapsed ? (
        <button
          onClick={() => setCollapsed((c) => !c)}
          className="fixed top-4 left-4 z-20 p-3 bg-[#F1F1F1] rounded text-2xl text-[#26324F] hover:text-[#1D2C4C] transition-colors"
        >
          ≡
        </button>
      ) : (
        <div className="flex flex-col bg-[#F1F1F1] border-r border-neutral-200 w-60">
          <button
            onClick={() => setCollapsed((c) => !c)}
            className="p-3 self-end text-2xl text-[#26324F] hover:text-[#1D2C4C] transition-colors"
          >
            ≡
          </button>
          <div className="px-4 py-2 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-[#26324F]">Generated Files</h2>
          </div>
          <div className="flex-1 overflow-y-auto px-2 space-y-1">
            {files.length === 0 ? (
              <div className="text-sm text-neutral-500 italic px-2 py-4">No files generated yet</div>
            ) : (
              files
                .slice()
                .sort((a, b) => new Date(b.created_at || 0) - new Date(a.created_at || 0))
                .map(renderFileItem)
            )}
          </div>
        </div>
      )}

      {/* Main */}
      <div className="flex flex-col flex-1">
        <header className="bg-transparent text-[#26324F] text-center py-6 text-4xl font-bold">
          SpillSense
        </header>

        {/* Quick Actions */}
        <div className="px-4 pb-2">
          <div className="flex flex-wrap gap-2 justify-center">
            {quickActions.map((a, i) => (
              <button
                key={i}
                onClick={() => handleQuickAction(a.query)}
                className="bg-white/80 hover:bg-white text-[#26324F] text-sm px-3 py-1 rounded-full border border-neutral-300 transition-all hover:shadow-md"
                disabled={sending}
              >
                {a.label}
              </button>
            ))}
          </div>
        </div>

        {/* Chat */}
        <div className="flex-1 overflow-y-auto p-4 flex flex-col space-y-3">
          {messages.map(renderMessage)}
          {sending && (
            <div className="text-sm text-neutral-500 italic self-center flex items-center gap-2">
              {messages.some((m) => m.id === THINKING_ID) ? '🤔 Analyzing insights across all pipelines...' : 'Processing…'}
              <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-neutral-500" />
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* Input */}
        <div className="flex items-center p-4 border-t border-neutral-200 space-x-2 bg-neutral-50/80">
          <input
            type="text"
            placeholder="Ask questions or type commands..."
            className="flex-1 border border-neutral-300 rounded-full px-4 py-2 focus:outline-none focus:ring-2 focus:ring-[#94A4CB] disabled:opacity-50"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && !sending && handleSend()}
            disabled={sending}
          />

          <input
            id="file"
            type="file"
            multiple
            className="hidden"
            onChange={onFileChange}
            disabled={sending}
            accept=".csv,.jpg,.jpeg,.png,.pdf,.bmp,.json,.md"
          />
          <label
            htmlFor="file"
            className={`px-4 py-2 rounded-full cursor-pointer transition-colors text-white ${
              sending ? 'bg-gray-400 cursor-not-allowed' : 'bg-[#7D94CD] hover:bg-[#6A82BB]'
            }`}
          >
            Upload
          </label>

          {selectedFiles.map((file, idx) => (
            <div key={idx} className="flex items-center space-x-1 bg-white px-2 py-1 rounded-full border border-neutral-300">
              <span className="text-sm italic text-neutral-600 max-w-20 truncate">{file.name}</span>
              <button onClick={() => removeFile(idx)} className="text-neutral-500 hover:text-neutral-700" disabled={sending}>
                ✕
              </button>
            </div>
          ))}

          <button
            onClick={handleSend}
            className="bg-[#1D2C4C] hover:bg-[#162341] text-white px-6 py-2 rounded-full disabled:opacity-50 transition-colors"
            disabled={sending || (!input.trim() && selectedFiles.length === 0)}
          >
            {sending ? (
              <div className="flex items-center space-x-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
                <span>Sending...</span>
              </div>
            ) : (
              'Send'
            )}
          </button>
        </div>

        {/* Footer */}
        <footer className="p-4 text-center bg-neutral-50/80 text-xs text-neutral-600">
          <div className="flex items-center justify-center space-x-4 mb-2">
            <img src="/logo.png" alt="SpillSense logo" className="h-16" />
            {lastQuery && (
              <div>
                Last query: <span className="italic">"{lastQuery.substring(0, 50)}..."</span>
                {retryCount > 0 && <span className="ml-2 text-orange-600">(Retry #{retryCount})</span>}
              </div>
            )}
          </div>
          {files.length > 0 && (
            <div>{files.length} file{files.length !== 1 ? 's' : ''} generated • Ready for analysis</div>
          )}
        </footer>
      </div>
    </div>
  );
}
