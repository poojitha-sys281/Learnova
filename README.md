# 🎓 Learnova — AI-Powered Personalized Learning Assistant

Learnova is an AI-powered learning assistant that helps students understand study materials, ask questions from uploaded PDFs, generate summaries, and practice with automatically generated quizzes.

It combines **Generative AI, Retrieval-Augmented Generation (RAG), embeddings, and vector search** to provide answers grounded in the user's uploaded learning material.

---

## ✨ Features

- 💬 **AI Study Chat** — Ask questions and get explanations using Gemini.
- 📄 **PDF Learning** — Upload lecture notes, textbooks, or study materials.
- 🔎 **RAG-Based Question Answering** — Retrieves relevant content from uploaded documents before generating answers.
- 🧠 **Semantic Search** — Uses embeddings to find the most relevant document sections.
- 🗃️ **Vector Storage** — Stores document embeddings using ChromaDB.
- 📚 **Automatic Summarization** — Generate concise summaries from uploaded documents.
- 📝 **AI Quiz Generation** — Automatically generate multiple-choice questions from study material.
- 🎯 **Difficulty Levels** — Choose Beginner, Intermediate, or Advanced explanations.
- 📊 **Quiz Scoring** — Submit answers and receive an automatically calculated score.
- 📌 **Source Context** — View the document sections used to generate RAG answers.
- 🎨 **Streamlit Interface** — Simple and interactive learning-focused UI.

---

## 🧠 How Learnova Works

Learnova follows a Retrieval-Augmented Generation (RAG) pipeline:

```text
                Uploaded PDF
                     │
                     ▼
              Text Extraction
                     │
                     ▼
                 Chunking
                     │
                     ▼
                Embeddings
                     │
                     ▼
                ChromaDB
               Vector Store
                     │
                     ▼
              Similarity Search
                     │
                     ▼
             Relevant Chunks
                     │
                     ▼
                Gemini LLM
                     │
                     ▼
             Learnova Answer
