# 🎓 Learnova

### AI-Powered Personalized Learning Assistant

Learnova is a Generative AI-powered learning assistant that helps students understand their study material through **PDF-based question answering, semantic search, RAG-powered responses, AI summaries, personalized explanations, and automatic MCQ generation**.

---

## ✨ Features

### 📚 PDF-Based Learning
Upload your study material as a PDF and interact with its contents using natural language.

### 🔎 Semantic Search
Learnova uses embeddings to understand the meaning of a user's question and retrieve the most relevant sections from the uploaded document.

### 🧠 RAG-Powered Answers
Learnova combines retrieved document context with Google's Gemini model to generate answers grounded in the uploaded study material.

### 📝 AI Summaries
Generate concise summaries of the uploaded study material using Generative AI.

### 🎯 Personalized Explanations
Choose your learning level:

- Beginner
- Intermediate
- Advanced

Learnova adjusts the explanation style according to the selected level.

### ❓ Automatic MCQ Generation
Generate multiple-choice questions automatically from the uploaded study material.

### 📊 Quiz Scoring
Attempt generated quizzes and receive your score instantly.

### 💬 AI Study Chat
Ask questions naturally and interact with Learnova as a personal study assistant.

---

# 🏗️ System Architecture

```text
                         ┌─────────────────────┐
                         │    Learnova UI      │
                         │     Streamlit       │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │     PDF Upload      │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │    Text Extraction  │
                         │      PyMuPDF        │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │      Chunking       │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │     Embeddings      │
                         │ SentenceTransformers│
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │      ChromaDB       │
                         │    Vector Store     │
                         └──────────┬──────────┘
                                    │
                              User Question
                                    │
                         ┌──────────▼──────────┐
                         │   Semantic Search   │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │  Relevant Context   │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │       Gemini        │
                         │      GenAI LLM      │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │   RAG-Based Answer  │
                         └─────────────────────┘