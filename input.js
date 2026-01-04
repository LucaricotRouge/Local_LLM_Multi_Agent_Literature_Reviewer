import { pipeline } from '@huggingface/transformers';
import { CreateMLCEngine } from '@mlc-ai/web-llm';
// import * as webllm from "https://esm.run/@mlc-ai/web-llm";

const state = {
  vectorStore: [], // { text, embedding: null, source }
  chunksCount: 0,
  chatHistory: [], // [{ role: 'user'|'assistant', content: string }]
  pdfCount: 0,
};

const systemPrompt = `You are an Academic Research Assistant. Answer succinctly, cite relevant parts of the provided context, and avoid fabricating references. If the context is insufficient, say so and suggest what to upload.`;

const llm = {
  engine: null,
  ready: false,
};

// Retrieval settings
// const numberOfRelevantChunks = 3; // ranking seed; final inclusion uses a token budget

// --- Token estimation helpers (rough: ~4 chars/token) ---
function tokensFromString(str) { return Math.ceil((str || '').length / 4); } // for brut string (chunks, resume, question), (/4 is average char per token)
// function tokensFromMsg(msg) { return 2 + tokensFromString(msg?.content || ''); } // for structured message like { role: 'user', content: '...' } (+2 is for role/content overhead)

const CONTEXT_TOKEN_BUDGET = 1500; // Limit injected context by token budget

const MAX_RECENT_TURNS = 3; // Keep at most N recent user-assistant turns in raw form => make sure recent context is preserved and do not exceed model input limits

function selectChunksByBudget(scored, budgetTokens) {
  const picked = [];
  let remaining = budgetTokens;
  let i = 0;
  for (const s of scored) {
    const snippet = (s.entry.text || '').slice(0, 500); // snippet is first 500 chars of the chunk
    const block = `[#${i + 1}] Source: ${s.entry.source}\n${snippet}`;
    const cost = tokensFromString(block) + 6; // small header overhead
    if (cost > remaining) break;
    picked.push({ block, scored: s });
    remaining -= cost;
    i++;
  }
  return picked;
}

function lastTurns(history, turns = MAX_RECENT_TURNS) {
  if (!Array.isArray(history) || history.length === 0) return [];
  const kept = [];
  let userTurns = 0;
  for (let i = history.length - 1; i >= 0; i--) {
    const m = history[i];
    if (m.role === 'user') userTurns++;
    kept.push(m);
    if (userTurns >= turns) break;
  }
  return kept.reverse();
}

async function summarizeHistory(messages) {
  if (!messages || !messages.length || !llm.ready || !llm.engine) return '';
  const transcript = messages.map(m => `${m.role}: ${m.content}`).join('\n').slice(0, 4000);
  const prompt = [
    { role: 'system', content: 'You are a summarizer. Create a concise, factual summary of prior chat: key questions, answers, constraints and decisions. Do not invent content.' },
    { role: 'user', content: transcript }
  ];
  try {
    const res = await llm.engine.chat.completions.create({ messages: prompt, temperature: 0.0 });
    return res?.choices?.[0]?.message?.content || '';
  } catch (e) {
    console.warn('Summarization failed:', e);
    return '';
  }
}

/**
 * Split text into overlapping character-based chunks using a sliding window.
 * @param {string} text - The input text to chunk.
 * @param {number} size - Target chunk size in characters (default 500).
 * @param {number} overlap - Overlap between consecutive chunks in characters (default 100).
 * @returns {string[]} Array of chunk strings.
 */

async function extractTextFromPDF(file, onProgress) { // file is a File object (html input or drag-drop : local file of the user)
  const arrayBuffer = await file.arrayBuffer(); // arrayBuffer() converts File to binary data
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise; // load PDF document from binary data, .promise allows await of all loading 
  // it contains : 
  // {
  // numPages: 12,
  // getPage: function(...)
  // }
  let fullText = '';
  if (typeof onProgress === 'function') onProgress(0);
  for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) { // for each page of pdf
    const page = await pdf.getPage(pageNum); // get the page
    const content = await page.getTextContent();
    // get the text content of the page as : 
    // {
    //   items: [
    //     { str: "Hello", transform: [...], width: ... },
    //     { str: "world", transform: [...], width: ... }, 
    //     { image: ..., transform: [...], width: ... }
    //   ]
    // }

    const strings = content.items.map((item) => item.str); // extract only the text strings
    fullText += strings.join(' ') + '\n'; // join them with space and add a newline at the end of the page
    if (typeof onProgress === 'function') {
      const pct = Math.round((pageNum / pdf.numPages) * 100);
      onProgress(pct);
    }
  }
  return fullText;
}

function chunkTextSliding(text, size = 500, overlap = 100) {
  if (typeof text !== 'string') return [];
  const cleanText = text; // No special cleaning for now (but needed for normalization and space suppression)
  const chunks = [];
  const s = Math.max(1, Math.floor(size)); // Ensure size is at least 1 and an integer
  const o = Math.min(Math.max(0, Math.floor(overlap)), s - 1); // Ensure overlap is not negative or >= size
  if (!cleanText.length) return chunks;
  const step = s - o; // Step size between chunks
  for (let i = 0; i < cleanText.length; i += step) { 
    const slice = cleanText.slice(i, i + s);
    if (slice.length) chunks.push(slice);
  }
  if (chunks.length > 1) {
    const last = chunks[chunks.length - 1];
    if (last.length < Math.max(1, Math.floor(s * 0.25))) {
      // If the last chunk is too short (< 25% of size), merge it into the previous chunk
      chunks[chunks.length - 2] = chunks[chunks.length - 2] + last;
      chunks.pop();
    }
  }
  return chunks;
}

// function updateChunkCount() {
//   const el = document.getElementById('chunk-count');
//   if (el) el.textContent = String(state.chunksCount);
// }

// function appendExtractedOutput(text, sourceName) {
//   const output = document.getElementById('extracted-output');
//   if (!output) return;
//   const header = `\n--- Fichier: ${sourceName} ---\n`;
//   output.textContent += header + text + '\n';
// }

let _extractorPromise = null; 
async function embedText(text) { // have to be initialized only once
  if (!_extractorPromise) {
    const WebGPU = typeof navigator !== 'undefined' && !!navigator.gpu;
    _extractorPromise = pipeline(
      'feature-extraction',
      'Xenova/all-MiniLM-L6-v2',
      { device: WebGPU ? 'webgpu' : 'cpu' }
    );
  }
  const extractor = await _extractorPromise;
  const result = await extractor(text, { pooling: 'mean', normalize: true });
  return result.data;
}

// Render chat history into the chat container
function renderChat() {
  const container = document.getElementById('chat-history');
  if (!container) return;
  container.textContent = '';
  for (const msg of state.chatHistory) {
    const div = document.createElement('div');
    div.className = msg.role === 'user' ? 'mb-2 text-black' : 'mb-2 text-blue-700';
    div.textContent = `${msg.role === 'user' ? 'You' : 'Assistant'}: ${msg.content}`;
    container.appendChild(div);
  }
  container.scrollTop = container.scrollHeight;
}

// Render citations from scored retrieval results
function renderCitations(scored) {
  const el = document.getElementById('citations');
  if (!el) return;
  if (!scored || !scored.length) { el.textContent = ''; return; }
  const items = scored.map((s) => {
    const scoreStr = Number.isFinite(s.score) ? s.score.toFixed(3) : 'n/a';
    return `• ${s.entry.source} (score: ${scoreStr})`;
  }).join('\n');
  el.textContent = `Citations used:\n${items}`;
}

async function initWebLLM() {
  const modelStatus = document.getElementById('model-status-text');
  try {
    if (modelStatus) modelStatus.textContent = 'Loading WebLLM…';
    // Let WebLLM choose the best available backend automatically
    llm.engine = await CreateMLCEngine('Llama-3.2-1B-Instruct-q4f32_1-MLC');
    llm.ready = true;
    if (modelStatus) modelStatus.textContent = 'Ready: Llama-3.2-1B-Instruct';
  } catch (err) {
    if (modelStatus) modelStatus.textContent = 'Failed to load model';
    console.error('WebLLM init error:', err);
  }
}

function cosineSimilarity(a, b) {
  if (!a || !b) return -Infinity;
  const len = Math.min(a.length, b.length);
  let dot = 0, dot_a = 0, dot_b = 0;
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
    dot_a += a[i] * a[i];
    dot_b += b[i] * b[i];
  }
  if (dot_a === 0 || dot_b === 0) return 0;
  return dot / Math.sqrt(dot_a * dot_b);
}

async function processPDFFile(file) {
  const text = await extractTextFromPDF(file, (pct) => {
    updateLoadingBar(pct);
  });
  const chunks = chunkTextSliding(text, 500, 100);
  const entries = chunks.map((chunkText, idx) => ({
    text: chunkText,
    embedding: null,
    source: `${file.name}#${idx + 1}`
  }));
  for (const entry of entries) {
    try {
      entry.embedding = await embedText(entry.text);
    } catch (e) {
      console.error('Failed to embed chunk', entry?.source, e);
      entry.embedding = null;
    }
  }
  state.vectorStore.push(...entries);
  state.chunksCount += chunks.length;
  // appendExtractedOutput(text, file.name);
  // updateChunkCount();
}

function setupDropZone() {
  const input = document.getElementById('pdf-input');
  const dropzone = document.getElementById('dropzone');

  if (dropzone && input) {   // Click to open file dialog

    dropzone.addEventListener('click', () => input.click());
  }

  if (input) {   // File input change

    input.addEventListener('change', async () => {
      const files = Array.from(input.files || []).filter((f) => (f.type === 'application/pdf') || /\.pdf$/i.test(f.name)); // allow PDFs by MIME or extension
      if (files.length) showLoadingBar();
      for (const f of files) {
        await processPDFFile(f);
        state.pdfCount += 1;
        updatePdfCountUI();
      }
      updateLoadingBar(100);
      hideLoadingBar();
      input.value = '';
    });
  }

  if (dropzone) {   // Drag-and-drop events on dropzone
    ['dragenter', 'dragover'].forEach((evt) => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault(); e.stopPropagation();
        dropzone.classList.add('bg-gray-50');
      });
    });
    ['dragleave', 'drop'].forEach((evt) => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault(); e.stopPropagation();
        dropzone.classList.remove('bg-gray-50');
      });
    });
    dropzone.addEventListener('drop', async (e) => {
      const files = Array.from(e.dataTransfer?.files || []).filter((f) => (f.type === 'application/pdf') || /\.pdf$/i.test(f.name));
      if (files.length) showLoadingBar();
      for (const f of files) {
        await processPDFFile(f);
        state.pdfCount += 1;
        updatePdfCountUI();
      }
      updateLoadingBar(100);
      hideLoadingBar();
    });
  }
}

async function handleQuery() {
  const prompt = document.getElementById('prompt-input');
  if (!prompt) return;
  const question = prompt.value.trim();
  if (!question) return;
  let scored = [];
  try {
    const qEmbedding = await embedText(question);
    scored = state.vectorStore.map((entry) => ({
      entry,
      score: cosineSimilarity(qEmbedding, entry.embedding),
    })).sort((a, b) => b.score - a.score); // Rank all; budget will cap inclusions
  } catch(err) {
    console.error('Embedding/query error:', err);
  }

  const picked = selectChunksByBudget(scored, CONTEXT_TOKEN_BUDGET);
  const contextBlock = picked.map(p => p.block).join('\n\n');

  state.chatHistory.push({ role: 'user', content: question });   // Update chat history with user message

  renderChat();
  renderCitations(picked.map(p => p.scored));

  if (!llm.ready || !llm.engine) {
    state.chatHistory.push({ role: 'assistant', content: 'Model not ready yet. Please wait for loading to finish.' });
    renderChat();
    return;
  }

  // Compose messages: system + summarized older history + last N turns + current turn
  const prior = state.chatHistory.slice(0, -1);
  const recent = lastTurns(prior, MAX_RECENT_TURNS);
  const older = prior.slice(0, Math.max(0, prior.length - recent.length));
  let summary = '';
  if (older.length) summary = await summarizeHistory(older);
  // WebLLM requires the system message to be the first and only system entry
  const systemContent = summary
    ? `${systemPrompt}\n\nConversation summary:\n${summary}`
    : systemPrompt;
  const finalPromptForTheLLM = [
    { role: 'system', content: systemContent },
    ...recent,
    { role: 'user', content: `Context (may be partial):\n${contextBlock || '(no context available)'}\n\nQuestion: ${question}` }
  ];

  const submit = document.getElementById('submit-prompt');
  if (submit) submit.disabled = true;
  try {
    const completion = await llm.engine.chat.completions.create({
      messages:finalPromptForTheLLM,
      temperature: 0.2,
    });
    const content = completion?.choices?.[0]?.message?.content || 'No response.';
    state.chatHistory.push({ role: 'assistant', content });
  } catch (err) {
    console.error('Chat error:', err);
    state.chatHistory.push({ role: 'assistant', content: 'An error occurred while generating the answer.' });
  } finally {
    if (submit) submit.disabled = false;
    renderChat();
    prompt.value = '';
  }
}

window.addEventListener('DOMContentLoaded', () => {
  setupDropZone();
  initWebLLM();
  const submit = document.getElementById('submit-prompt');
  if (submit) submit.addEventListener('click', handleQuery);
});









function createLoadingBar() {
  const wrapper = document.createElement('div');
  wrapper.style.border = '4px solid black';
  wrapper.style.borderRadius = '12px';
  wrapper.style.height = '24px';
  wrapper.style.width = '100%';
  wrapper.style.position = 'relative';
  wrapper.style.overflow = 'hidden';

  const fill = document.createElement('div');
  fill.style.height = '100%';
  fill.style.width = '0%';
  fill.style.background = 'rgb(172, 50, 50)';
  fill.style.borderRadius = '12px';

  const label = document.createElement('span');
  label.style.position = 'absolute';
  label.style.left = '50%';
  label.style.top = '50%';
  label.style.transform = 'translate(-50%, -50%)';
  label.style.fontFamily = 'Hack, monospace';
  label.style.fontSize = '14px';
  label.textContent = '0%';

  wrapper.appendChild(fill);
  wrapper.appendChild(label);

  return {
    element: wrapper,
    update: (pct) => {
      const clamped = Math.max(0, Math.min(100, Math.round(pct)));
      fill.style.width = clamped + '%';
      label.textContent = clamped + '%';
    }
  };
}

function getDropzone() {
  return document.getElementById('dropzone');
}

function getHintEl() {
  const dz = getDropzone();
  return dz ? dz.querySelector('p.text-lg') : null;
}

function getAcceptedTypesEl() {
  const dz = getDropzone();
  if (!dz) return null;
  const items = dz.querySelectorAll('p');
  // try to find the accepted types line
  for (const p of items) {
    if (p.textContent && p.textContent.toLowerCase().includes('accepted') && p.textContent.toLowerCase().includes('pdf')) return p;
  }
  return null;
}

function ensurePdfCountEl() {
  const parent = getDropzone();
  const anchor = getAcceptedTypesEl();
  if (!parent || !anchor) return null;
  let counter = parent.querySelector('#pdf-count');
  if (!counter) {
    counter = document.createElement('p');
    counter.id = 'pdf-count';
    counter.className = 'mt-2 text-sm';
    counter.textContent = `Loaded PDFs: ${state.pdfCount}`;
    anchor.insertAdjacentElement('afterend', counter);
  }
  return counter;
}

function updatePdfCountUI() {
  const el = ensurePdfCountEl();
  if (el) el.textContent = `Loaded PDFs: ${state.pdfCount}`;
}

let loadingBarRef = null;

function showLoadingBar() {
  if (loadingBarRef) return loadingBarRef;
  const hint = getHintEl();
  const dz = getDropzone();
  if (!dz) return null;
  const bar = createLoadingBar();
  if (hint) {
    // store original hint to dataset for later restore
    if (!dz.dataset.originalHintText) dz.dataset.originalHintText = hint.textContent || '';
    hint.replaceWith(bar.element);
  } else {
    dz.prepend(bar.element);
  }
  loadingBarRef = bar;
  return loadingBarRef;
}

function updateLoadingBar(pct) {
  if (loadingBarRef) loadingBarRef.update(pct);
}

function hideLoadingBar() {
  const dz = getDropzone();
  if (!dz || !loadingBarRef) return;
  const originalText = dz.dataset.originalHintText || 'Drag and drop your PDF here or click to select';
  const p = document.createElement('p');
  p.className = 'text-lg';
  p.textContent = originalText;
  loadingBarRef.element.replaceWith(p);
  loadingBarRef = null;
}

// // Embeddings: initialize lazy pipeline and helpers
// let embedderPromise = null;
// function getEmbedder() {
//   if (!embedderPromise) {
//     embedderPromise = pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
//   }
//   return embedderPromise;
// }

