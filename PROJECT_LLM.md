# Project: Local LLM Multi-Agent Literature Reviewer 📚

## Project Goal

The objective of this project is to build a privacy-first, fully browser-based AI research assistant. Unlike cloud-based solutions, this application will run entirely on the client side using **WebLLM** (for reasoning) and **Transformers.js** (for embeddings and audio).

You will build a system that can ingest PDF research papers, index them into a custom client-side Vector Database, and use an LLM Agent to synthesize a high-quality Literature Review based on your chat queries and the provided documents.

**To Pass the Project, a final score of** $\ge 10$ **is expected for each section.**

## 1. RAG Engine: Processing & Vector Store

| Weight: 30% (Difficulty: High) |
| :--- |

This section focuses on the "reading" capability of your agent. You must implement a Retrieval Augmented Generation (RAG) pipeline entirely in the browser using Vanilla JavaScript.

### 1.1. PDF Extraction & Chunking

1.  **Ingestion:** Implement a drag-and-drop zone using HTML/JS to accept PDF files.
2.  **Extraction:** Use the `pdf.js` library (via CDN) to extract raw text from uploaded documents.
3.  **Chunking:** Implement a JavaScript function to split the text into manageable chunks (e.g., sliding window of 500 characters with 100 character overlap).

### 1.2. Embeddings & Custom Vector DB

1.  **Transformers.js:** Initialize the pipeline using `Xenova/all-MiniLM-L6-v2` to generate numerical embeddings (vectors) for each text chunk.
2.  **Vector Store:** Create a custom global array (e.g., `state.vectorStore`) to store the chunks. Each entry should contain:
    * `text`: The raw text content.
    * `embedding`: The vector data.
    * `source`: Filename or metadata.
3.  **Search Algorithm:** Implement a **Cosine Similarity** function in pure JavaScript to find the most relevant chunks given a user query vector.
    $$\text{similarity} = \frac{\sum (A_i \times B_i)}{\sqrt{\sum A_i^2} \times \sqrt{\sum B_i^2}}$$

| Grading Scale (0-20) | Description |
| :--- | :--- |
| **0** | No PDF processing or embedding implementation. |
| **10 (Minimum Goal)** | Can upload a PDF and extract raw text. The text is passed directly to the LLM context window (naive approach) without embeddings/search. |
| **15 (Good Goal)** | Text is chunked and embedded using Transformers.js. A custom `cosineSimilarity` function returns relevant chunks for a hardcoded query. |
| **20 (Excellent Goal)** | A robust Vector DB is implemented. The system handles multiple PDFs simultaneously. The UI visualizes the "Memory Bank" (showing chunk counts or storage usage) as seen in the reference implementation. |

## 2. The Agentic Chat Interface

| Weight: 25% (Difficulty: Medium) |
| :--- |

This section covers the WebLLM integration and the user interface.

### 2.1. WebLLM Integration

1.  **Model Loading:** Integrate **WebLLM** (via CDN/ESM) and load a lightweight model (e.g., `Llama-3.2-1B-Instruct`) compatible with WebGPU.
2.  **Chat History:** Maintain a JavaScript array for conversation history and inject this into the prompt context so the agent "remembers" previous turns.

### 2.2. The "Lit Reviewer" Agent

1.  **System Prompting:** Craft a system prompt that instructs the model to act as an Academic Researcher.
2.  **Context Injection:** When the user asks a question:
    * Generate an embedding for the question.
    * Retrieve the top $N$ relevant chunks from your Vector Store.
    * Construct a prompt that includes: `System Instruction` + `Chat History` + `Retrieved Context` + `User Question`.
3.  **UI/UX:** Build a clean interface using **Tailwind CSS** (CDN). It must show:
    * Model loading status.
    * The chat history.
    * **Crucial:** A visual indicator of which document sections (citations) are being used for the answer.

| Grading Scale (0-20) | Description |
| :--- | :--- |
| **0** | No chat interface or LLM integration. |
| **10 (Minimum Goal)** | Basic chat works with WebLLM. No history (amnesia between messages). No RAG context injection. |
| **15 (Good Goal)** | Chat history works. RAG is integrated: The model answers questions using extracted context. UI is clean and responsive. |
| **20 (Excellent Goal)** | Full Agentic workflow. The user can request a "Literature Review," and the Agent autonomously aggregates summaries. The UI includes a "System Controls" panel to tweak Temperature and System Prompts in real-time. |

## 3. Subjective Quality: The Literature Review Output

| Weight: 20% (Difficulty: N/A - Quality Check) |
| :--- |

This section grades the *quality* of the AI's output during the live demo.
**Test Scenario:** The evaluator will chat briefly about a topic, upload **6 PDF research papers**, and ask the Agent to "Generate a concise literature review."

| Grading Scale (0-20) | Description |
| :--- | :--- |
| **0** | Output is incoherent, hallucinates papers not uploaded, or fails to generate. |
| **10 (Minimum Goal)** | The review mentions the uploaded papers but lacks depth. It reads like a list of abstracts. |
| **15 (Good Goal)** | The review synthesizes themes across papers (e.g., "While Paper A suggests X, Paper B argues Y"). Hallucinations are minimal. |
| **20 (Excellent Goal)** | Professional academic quality. The review is structured (Introduction, Comparison, Conclusion). The agent correctly references specific details/numbers from the vector store to back up claims. |

## 4. Deployment & Professional Documentation

| Weight: 15% (Difficulty: Low) |
| :--- |

### 4.1. GitHub Pages Deployment

* The application must be deployed as a **Single Page Application** on GitHub Pages.
* It must be accessible via a public URL.

### 4.2. Repository & README

* **Project Title & Visuals:** Screenshot of the interface with a generated Lit Review.
* **Tech Stack:** WebLLM, Transformers.js, Tailwind, Vanilla JS.
* **Usage Guide:** Clear instructions on how to run it locally.
* **Local Setup:** Mention the need for a local server (e.g., Python or VS Code Live Server) for development.

| Grading Scale (0-20) | Description |
| :--- | :--- |
| **0** | Not deployed. No README. |
| **10 (Minimum)** | Deployed but buggy. Basic README. |
| **15 (Good)** | Deployed and fully functional. README explains the "Local AI" privacy aspect clearly. |
| **20 (Very Good)** | Polished deployment. README includes a GIF of the RAG process. Code is clean, modular, and well-commented. |

## 5. Bonus: Multimodal Accessibility (Voice)

| Weight: 10% (Difficulty: High) |
| :--- |

Add voice interaction to your researcher agent using **Transformers.js**.

* **STT (Speech-to-Text):** Use `Xenova/whisper-tiny` to transcribe microphone input.
* **VAD (Voice Activity Detection):** Implement a logic (like measuring RMS amplitude) to detect when the user stops speaking to automatically submit the query.
* **TTS (Text-to-Speech):** Use the browser's native `SpeechSynthesis` API to read the response.

| Grading Scale (0-20) | Description |
| :--- | :--- |
| **0** | No audio features. |
| **10 (Minimum)** | Browser Native TTS is implemented (Agent can speak). |
| **15 (Good)** | STT is implemented using Whisper (User can speak to Agent). |
| **20 (Excellent)** | Full "Hands-Free" mode: VAD automatically detects when you stop talking, sends the query, and the Agent reads the response back. Visual feedback shows "Listening" vs "Processing" states. |

## Summary of Weighting and Passing Criteria

| Section | Description | Weight | Passing Score (Minimum) |
| :--- | :--- | :--- | :--- |
| **1. RAG Engine** | PDF & Vectors | 30% | $\ge 10/20$ |
| **2. Chat Interface** | WebLLM & UI | 25% | $\ge 10/20$ |
| **3. Output Quality** | Lit Review Test | 20% | $\ge 10/20$ |
| **4. Deployment** | GH Pages | 15% | $\ge 10/20$ |
| **5. Bonus** | Voice/Audio | 10% | Optional |
| **TOTAL** | | **100%** | **Expected Pass:** $\ge 10/20$ **on Sections 1-4** |

### Appendix: Technical Setup & Resources

#### A. Local Server for Testing 💻

Because this project uses ES Modules and loads external files, you cannot simply open `index.html` by double-clicking it. You must run a local server.

1.  Navigate to your project folder.
2.  Run Python's built-in server:
    ```bash
    python -m http.server
    ```
3.  Open `http://localhost:8000`.

*(Note: If you encounter "SharedArrayBuffer" errors, you may need to use a browser profile that disables security isolation for localhost, or use a dev server that supports COOP/COEP headers).*

#### B. Resources 📚

| Resource | Description | Link |
| :--- | :--- | :--- |
| **WebLLM** | High-performance in-browser LLM inference. | <https://webllm.mlc.ai/> |
| **Transformers.js** | Machine Learning for the web (Embeddings/Whisper). | <https://huggingface.co/docs/transformers.js/> |
| **PDF.js** | PDF parsing library. | <https://mozilla.github.io/pdf.js/> |
