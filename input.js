// Step 1: Ingestion + Step 2: Extraction (no embeddings yet)
const state = {
  vectorStore: [], // { text, embedding: null, source }
  chunksCount: 0,
};

async function extractTextFromPDF(file) {
  const arrayBuffer = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  let fullText = '';
  for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
    const page = await pdf.getPage(pageNum);
    const content = await page.getTextContent();
    const strings = content.items.map((item) => item.str);
    fullText += strings.join(' ') + '\n';
  }
  return fullText;
}

function updateChunkCount() {
  const el = document.getElementById('chunk-count');
  if (el) el.textContent = String(state.chunksCount);
}

function appendExtractedOutput(text, sourceName) {
  const output = document.getElementById('extracted-output');
  if (!output) return;
  const header = `\n--- Fichier: ${sourceName} ---\n`;
  output.textContent += header + text + '\n';
}

async function processPDFFile(file) {
  const text = await extractTextFromPDF(file);
  // Store the raw text; embedding will be added later in next steps
  state.vectorStore.push({ text, embedding: null, source: file.name });
  state.chunksCount += 1; // count per file for now
  appendExtractedOutput(text, file.name);
  updateChunkCount();
}

function setupDropZone() {
  const input = document.getElementById('pdf-input');
  const dropzone = document.getElementById('dropzone');

  // Click to open file dialog
  if (dropzone && input) {
    dropzone.addEventListener('click', () => input.click());
  }

  // File input change
  if (input) {
    input.addEventListener('change', async () => {
      const files = Array.from(input.files || []).filter((f) => f.type === 'application/pdf');
      for (const f of files) await processPDFFile(f);
      input.value = '';
    });
  }

  // Drag-and-drop events on dropzone
  if (dropzone) {
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
      const files = Array.from(e.dataTransfer?.files || []).filter((f) => f.type === 'application/pdf');
      for (const f of files) await processPDFFile(f);
    });
  }
}

async function handleQuery() {
  const ta = document.getElementById('prompt-input');
  if (!ta) return;
  const question = ta.value.trim();
  if (!question) return;
  const qEmbedding = await embedText(question);
  // Find top 3 similar chunks
  const scored = state.vectorStore.map((entry) => ({
    entry,
    score: cosineSimilarity(qEmbedding, entry.embedding),
  })).sort((a, b) => b.score - a.score).slice(0, 3);
  console.log('Top matches:', scored.map(s => ({ score: s.score.toFixed(4), source: s.entry.source, text: s.entry.text.slice(0, 200) })));
}

window.addEventListener('DOMContentLoaded', () => {
  setupDropZone();
});
