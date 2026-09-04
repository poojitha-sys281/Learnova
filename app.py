import os
import json
import time
import hashlib
import re

import streamlit as st
from google import genai
from dotenv import load_dotenv
import fitz
import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Learnova",
    page_icon="🎓",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    h1 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
    }

    h2, h3 {
        font-weight: 600 !important;
    }

    .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1rem;
    }

    [data-testid="stFileUploader"] {
        border-radius: 12px;
    }

    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 0.5rem;
    }

    [data-testid="stMetric"] {
        border-radius: 10px;
        padding: 10px;
    }

    [data-testid="stSidebar"] {
        padding-top: 1rem;
    }

    hr {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "quiz" not in st.session_state:
    st.session_state.quiz = []

if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

if "score" not in st.session_state:
    st.session_state.score = 0

if "answers" not in st.session_state:
    st.session_state.answers = {}

if "processed_file_hash" not in st.session_state:
    st.session_state.processed_file_hash = None

if "processed_file_name" not in st.session_state:
    st.session_state.processed_file_name = None

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "embedding_count" not in st.session_state:
    st.session_state.embedding_count = 0

if "summary" not in st.session_state:
    st.session_state.summary = ""

if "quiz_version" not in st.session_state:
    st.session_state.quiz_version = 0


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🎓 Learnova")

    st.markdown("### AI-Powered Learning Assistant")

    st.divider()

    st.markdown("### ✨ Features")

    st.markdown(
        """
        - 📄 PDF Upload
        - 🧠 RAG-based Q&A
        - 🔎 Semantic Search
        - 📝 AI Summarization
        - ❓ Automatic MCQs
        - 🎯 Difficulty Levels
        - 📊 Quiz Scoring
        - 💬 AI Study Chat
        """
    )

    st.divider()

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# API KEY CHECK
# ============================================================

if not api_key:

    st.error(
        "GEMINI_API_KEY is not configured. "
        "Please add it to your .env file or Streamlit Cloud Secrets."
    )

    st.stop()


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(api_key=api_key)


# ============================================================
# GEMINI GENERATION WITH RETRIES
# ============================================================

def generate_with_fallback(prompt):
    """
    Generate text using Gemini Interactions API.

    Uses retries for temporary service errors and
    falls back to another model if necessary.
    """

    models = [
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash"
    ]

    last_error = None

    for model in models:

        for attempt in range(3):

            try:

                response = client.interactions.create(
                    model=model,
                    input=prompt
                )

                if response and response.output_text:

                    return response.output_text.strip()

                last_error = Exception(
                    "Gemini returned an empty response."
                )

            except Exception as e:

                last_error = e

                error_text = str(e).upper()

                temporary_error = any(
                    code in error_text
                    for code in [
                        "503",
                        "500",
                        "429",
                        "UNAVAILABLE",
                        "SERVICE UNAVAILABLE",
                        "RESOURCE EXHAUSTED",
                        "INTERNAL"
                    ]
                )

                if temporary_error:

                    if attempt < 2:

                        wait_time = 2 ** attempt

                        time.sleep(wait_time)

                        continue

                break

    if last_error:
        print("Gemini Error:", last_error)

    return None


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ============================================================
# CHROMADB
# ============================================================

@st.cache_resource
def load_chroma_collection():

    chroma_client = chromadb.PersistentClient(
        path="./vectorstore"
    )

    collection = chroma_client.get_or_create_collection(
        name="learnova_documents"
    )

    return collection


collection = load_chroma_collection()


# ============================================================
# TEXT CHUNKING
# ============================================================

def chunk_text(
    text,
    chunk_size=1000,
    overlap=150
):
    """
    Split document text into overlapping chunks.
    """

    if not text:
        return []

    chunks = []

    start = 0

    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():

            chunks.append(
                chunk.strip()
            )

        if end >= text_length:
            break

        start = end - overlap

    return chunks


# ============================================================
# FILE HASH
# ============================================================

def get_file_hash(file_bytes):

    return hashlib.sha256(
        file_bytes
    ).hexdigest()


# ============================================================
# CLEAR CHROMADB COLLECTION
# ============================================================

def clear_collection():

    try:

        existing = collection.get()

        existing_ids = existing.get(
            "ids",
            []
        )

        if existing_ids:

            collection.delete(
                ids=existing_ids
            )

    except Exception as e:

        print(
            "Error clearing ChromaDB:",
            e
        )


# ============================================================
# GET ALL STORED DOCUMENTS
# ============================================================

def get_all_documents():

    try:

        results = collection.get(
            include=[
                "documents",
                "metadatas"
            ]
        )

        documents = results.get(
            "documents",
            []
        )

        metadatas = results.get(
            "metadatas",
            []
        )

        combined = list(
            zip(
                documents,
                metadatas
            )
        )

        combined.sort(
            key=lambda x: x[1].get(
                "chunk",
                0
            )
        )

        return combined

    except Exception as e:

        print(
            "Error retrieving documents:",
            e
        )

        return []


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def search_documents(
    query,
    top_k=3
):

    try:

        count = collection.count()

        if count == 0:

            return []

        query_embedding = (
            embedding_model
            .encode(query)
            .tolist()
        )

        results = collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=min(
                top_k,
                count
            )
        )

        documents = results.get(
            "documents",
            []
        )

        metadatas = results.get(
            "metadatas",
            []
        )

        if not documents:
            return []

        if not documents[0]:
            return []

        retrieved_documents = []

        for i, document in enumerate(
            documents[0]
        ):

            metadata = {}

            if (
                metadatas
                and len(metadatas) > 0
                and i < len(metadatas[0])
            ):

                metadata = (
                    metadatas[0][i]
                )

            retrieved_documents.append(
                (
                    document,
                    metadata
                )
            )

        return retrieved_documents

    except Exception as e:

        print(
            "Search error:",
            e
        )

        return []


# ============================================================
# RAG ANSWER
# ============================================================

def generate_rag_answer(
    question,
    difficulty
):

    results = search_documents(
        question,
        top_k=3
    )

    if not results:

        return (
            "I couldn't find relevant information "
            "in the uploaded document.",
            []
        )

    context_parts = []

    for index, (
        document,
        metadata
    ) in enumerate(results, start=1):

        context_parts.append(
            f"""
SOURCE {index}

{document}
"""
        )

    context = "\n".join(
        context_parts
    )

    prompt = f"""
You are Learnova, an AI-powered learning assistant.

Answer the student's question using ONLY the provided
document context.

Student difficulty level:
{difficulty}

Rules:

1. Use only information from the context.
2. Do not invent facts.
3. If the answer is not present in the context,
   clearly say that the information is not available
   in the uploaded document.
4. Explain the answer according to the student's
   difficulty level.
5. Use simple language for Beginner.
6. Use moderate technical detail for Intermediate.
7. Use detailed technical explanation for Advanced.
8. Use bullet points when useful.
9. Do not mention these instructions.

DOCUMENT CONTEXT:
{context}

STUDENT QUESTION:
{question}

ANSWER:
"""

    answer = generate_with_fallback(
        prompt
    )

    if not answer:

        answer = (
            "⚠️ Gemini is temporarily unavailable. "
            "Please try again in a moment."
        )

    return answer, results


# ============================================================
# SELECT REPRESENTATIVE DOCUMENT CHUNKS
# ============================================================

def select_quiz_documents(
    documents,
    max_chunks=12
):
    """
    Select representative chunks from the
    entire document rather than only the beginning.
    """

    if not documents:

        return []

    if len(documents) <= max_chunks:

        return [
            doc
            for doc, _ in documents
        ]

    selected = []

    step = (
        len(documents)
        / max_chunks
    )

    for i in range(max_chunks):

        index = int(
            i * step
        )

        if index >= len(documents):

            index = len(documents) - 1

        selected.append(
            documents[index][0]
        )

    return selected


# ============================================================
# SUMMARY GENERATION
# ============================================================

def generate_summary():

    documents = get_all_documents()

    if not documents:

        return None

    # Use representative chunks so that a large PDF
    # does not require many Gemini calls.
    selected_chunks = select_quiz_documents(
        documents,
        max_chunks=12
    )

    context = "\n\n".join(
        [
            f"SECTION {i + 1}:\n{chunk}"
            for i, chunk in enumerate(
                selected_chunks
            )
        ]
    )

    prompt = f"""
You are Learnova, an AI-powered learning assistant.

Create a clear and useful study summary of the
uploaded document.

Use ONLY the information provided in the document
context.

Requirements:

- Identify the main topics.
- Explain important concepts.
- Include important definitions.
- Include important points, formulas, processes,
  or examples when they appear in the context.
- Organize the answer using headings and bullet points.
- Make it useful for exam preparation.
- Do not invent information.
- Do not mention that only selected sections were used.
- Keep the summary concise but informative.

DOCUMENT CONTEXT:

{context}

STUDY SUMMARY:
"""

    return generate_with_fallback(
        prompt
    )


# ============================================================
# EXTRACT JSON FROM GEMINI RESPONSE
# ============================================================

def extract_json(text):

    if not text:

        return None

    text = text.strip()

    # Remove Markdown code fences
    text = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```",
        "",
        text
    )

    text = text.strip()

    # Direct JSON parsing
    try:

        return json.loads(text)

    except json.JSONDecodeError:

        pass

    # Find JSON array inside response
    start = text.find("[")

    end = text.rfind("]")

    if start != -1 and end != -1:

        json_text = text[
            start:end + 1
        ]

        try:

            return json.loads(
                json_text
            )

        except json.JSONDecodeError:

            return None

    return None


# ============================================================
# VALIDATE QUIZ
# ============================================================

def validate_quiz(quiz):

    if not isinstance(
        quiz,
        list
    ):

        return False

    if len(quiz) < 5:

        return False

    for item in quiz:

        if not isinstance(
            item,
            dict
        ):

            return False

        required_keys = [
            "question",
            "options",
            "answer",
            "explanation"
        ]

        for key in required_keys:

            if key not in item:

                return False

        options = item["options"]

        if not isinstance(
            options,
            list
        ):

            return False

        if len(options) != 4:

            return False

        if item["answer"] not in options:

            return False

    return True


# ============================================================
# GENERATE QUIZ
# ============================================================

def generate_quiz(
    difficulty
):

    documents = get_all_documents()

    if not documents:

        return None

    selected_chunks = select_quiz_documents(
        documents,
        max_chunks=12
    )

    context = "\n\n".join(
        [
            f"SECTION {i + 1}:\n{chunk}"
            for i, chunk in enumerate(
                selected_chunks
            )
        ]
    )

    prompt = f"""
You are Learnova, an AI-powered learning assistant.

Generate exactly 5 multiple-choice questions
from the provided document context.

Student difficulty level:
{difficulty}

Difficulty rules:

Beginner:
- Basic concepts
- Definitions
- Simple understanding

Intermediate:
- Conceptual understanding
- Application
- Comparisons

Advanced:
- Deeper concepts
- Reasoning
- Application
- Technical understanding

IMPORTANT:

Return ONLY valid JSON.

Do NOT use Markdown.
Do NOT use ```json.
Do NOT add explanations outside the JSON.

The JSON must have exactly this structure:

[
  {{
    "question": "Question text",
    "options": [
      "Option A",
      "Option B",
      "Option C",
      "Option D"
    ],
    "answer": "Correct option",
    "explanation": "Short explanation"
  }}
]

Rules:

1. Generate exactly 5 questions.
2. Each question must have exactly 4 options.
3. The answer must exactly match one of the options.
4. Each question must be based on the provided context.
5. Do not invent information.
6. Avoid duplicate questions.
7. Make the correct answer position vary between questions.

DOCUMENT CONTEXT:

{context}
"""

    response = generate_with_fallback(
        prompt
    )

    if not response:

        return None

    quiz = extract_json(
        response
    )

    if not validate_quiz(
        quiz
    ):

        return None

    return quiz[:5]


# ============================================================
# PDF PROCESSING
# ============================================================

def process_pdf(uploaded_file):

    file_bytes = uploaded_file.getvalue()

    file_hash = get_file_hash(
        file_bytes
    )

    # Do not process the same PDF repeatedly
    if (
        st.session_state.processed_file_hash
        == file_hash
    ):

        return

    # Clear previous document data
    clear_collection()

    # Reset generated content
    st.session_state.summary = ""

    st.session_state.quiz = []

    st.session_state.quiz_submitted = False

    st.session_state.score = 0

    st.session_state.answers = {}

    # Extract PDF text
    try:

        doc = fitz.open(
            stream=file_bytes,
            filetype="pdf"
        )

        text_parts = []

        for page in doc:

            page_text = page.get_text()

            if page_text.strip():

                text_parts.append(
                    page_text
                )

        doc.close()

        text = "\n".join(
            text_parts
        )

    except Exception as e:

        st.error(
            f"Could not read the PDF: {e}"
        )

        return

    if not text.strip():

        st.error(
            "No readable text was found in this PDF."
        )

        return

    # Chunk document
    chunks = chunk_text(
        text,
        chunk_size=1000,
        overlap=150
    )

    if not chunks:

        st.error(
            "Could not create document chunks."
        )

        return

    # Generate embeddings
    try:

        embeddings = embedding_model.encode(
            chunks
        )

        embeddings = embeddings.tolist()

    except Exception as e:

        st.error(
            f"Embedding generation failed: {e}"
        )

        return

    # Create unique IDs
    ids = []

    for i, chunk in enumerate(chunks):

        chunk_hash = hashlib.md5(
            chunk.encode(
                "utf-8"
            )
        ).hexdigest()[:10]

        ids.append(
            f"{file_hash[:12]}_{i}_{chunk_hash}"
        )

    # Metadata
    metadatas = []

    for i in range(len(chunks)):

        metadatas.append(
            {
                "source": uploaded_file.name,
                "chunk": i
            }
        )

    # Store in ChromaDB
    try:

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

    except Exception as e:

        st.error(
            f"ChromaDB storage failed: {e}"
        )

        return

    # Save session information
    st.session_state.processed_file_hash = (
        file_hash
    )

    st.session_state.processed_file_name = (
        uploaded_file.name
    )

    st.session_state.pdf_text = text

    st.session_state.chunk_count = len(
        chunks
    )

    st.session_state.embedding_count = len(
        embeddings
    )

    # Clear previous chat because document changed
    st.session_state.messages = []


# ============================================================
# MAIN UI
# ============================================================

st.title("🎓 Learnova")

st.caption(
    "Your AI-powered personalized learning assistant"
)


# ============================================================
# DIFFICULTY LEVEL
# ============================================================

difficulty = st.selectbox(
    "🎯 Choose your learning level",
    [
        "Beginner",
        "Intermediate",
        "Advanced"
    ]
)


# ============================================================
# PDF UPLOAD
# ============================================================

st.subheader(
    "📄 Upload your study material"
)

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"],
    help="Upload lecture notes, textbooks, study material, or question papers."
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file:

    current_hash = get_file_hash(
        uploaded_file.getvalue()
    )

    if (
        st.session_state.processed_file_hash
        != current_hash
    ):

        with st.spinner(
            "Processing your PDF..."
        ):

            process_pdf(
                uploaded_file
            )

        if (
            st.session_state.processed_file_hash
            == current_hash
        ):

            st.success(
                "✅ PDF processed successfully!"
            )

    else:

        st.success(
            f"📚 {uploaded_file.name} is ready."
        )


# ============================================================
# DOCUMENT STATUS
# ============================================================

if st.session_state.processed_file_hash:

    st.divider()

    st.subheader(
        "📊 Document Status"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Chunks",
            st.session_state.chunk_count
        )

    with col2:

        st.metric(
            "Embeddings",
            st.session_state.embedding_count
        )

    with col3:

        st.metric(
            "Difficulty",
            difficulty
        )


# ============================================================
# EXTRACTED TEXT
# ============================================================

if st.session_state.pdf_text:

    with st.expander(
        "🔍 View extracted text"
    ):

        st.text(
            st.session_state.pdf_text[:10000]
        )


# ============================================================
# SUMMARY
# ============================================================

if st.session_state.processed_file_hash:

    st.divider()

    st.subheader(
        "📝 AI Study Summary"
    )

    if st.button(
        "✨ Generate Summary"
    ):

        with st.spinner(
            "Learnova is creating your summary..."
        ):

            summary = generate_summary()

        if summary:

            st.session_state.summary = summary

        else:

            st.error(
                "⚠️ Gemini is temporarily unavailable. "
                "Please try again in a moment."
            )

    if st.session_state.summary:

        st.markdown(
            st.session_state.summary
        )


# ============================================================
# QUIZ GENERATION
# ============================================================

if st.session_state.processed_file_hash:

    st.divider()

    st.subheader(
        "❓ AI Quiz"
    )

    if st.button(
        "🎯 Generate 5 MCQs"
    ):

        with st.spinner(
            "Creating your quiz..."
        ):

            quiz = generate_quiz(
                difficulty
            )

        if quiz:

            st.session_state.quiz = quiz

            st.session_state.quiz_submitted = (
                False
            )

            st.session_state.score = 0

            st.session_state.answers = {}

            st.session_state.quiz_version += 1

        else:

            st.error(
                "⚠️ Could not generate the quiz. "
                "Please try again."
            )


# ============================================================
# DISPLAY QUIZ
# ============================================================

if st.session_state.quiz:

    st.markdown(
        f"### 🎯 {difficulty} Quiz"
    )

    for i, question in enumerate(
        st.session_state.quiz
    ):

        st.markdown(
            f"**Q{i + 1}. {question['question']}**"
        )

        answer = st.radio(
            "Choose an answer:",
            question["options"],
            key=f"quiz_{st.session_state.quiz_version}_{i}",
            index=None
        )

        st.session_state.answers[i] = answer

        st.write("")

    if st.button(
        "📊 Submit Quiz"
    ):

        score = 0

        for i, question in enumerate(
            st.session_state.quiz
        ):

            selected = (
                st.session_state.answers
                .get(i)
            )

            if (
                selected
                == question["answer"]
            ):

                score += 1

        st.session_state.score = score

        st.session_state.quiz_submitted = (
            True
        )

    # ========================================================
    # QUIZ RESULTS
    # ========================================================

    if st.session_state.quiz_submitted:

        score = st.session_state.score

        total = len(
            st.session_state.quiz
        )

        percentage = (
            score / total
        ) * 100

        st.success(
            f"🎉 Your Score: {score}/{total} "
            f"({percentage:.0f}%)"
        )

        st.markdown(
            "### 📖 Answer Review"
        )

        for i, question in enumerate(
            st.session_state.quiz
        ):

            selected = (
                st.session_state.answers
                .get(i)
            )

            correct = (
                selected
                == question["answer"]
            )

            if correct:

                st.success(
                    f"Q{i + 1}: Correct ✅"
                )

            else:

                st.error(
                    f"Q{i + 1}: Incorrect ❌"
                )

                st.write(
                    f"Your answer: "
                    f"{selected or 'Not answered'}"
                )

                st.write(
                    f"Correct answer: "
                    f"{question['answer']}"
                )

            st.caption(
                question["explanation"]
            )


# ============================================================
# AI STUDY CHAT
# ============================================================

st.divider()

st.subheader(
    "💬 Ask Learnova"
)

if not st.session_state.processed_file_hash:

    st.info(
        "📄 Upload a PDF first, then ask questions "
        "about your study material."
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"]
            == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "🔎 View retrieved sources"
            ):

                for source in message["sources"]:

                    st.markdown(
                        f"""
                        **Chunk {source["chunk"] + 1}**

                        {source["text"]}
                        """
                    )


# ============================================================
# CHAT INPUT
# ============================================================

if st.session_state.processed_file_hash:

    prompt = st.chat_input(
        "Ask a question about your PDF..."
    )

    if prompt:

        # Display user message
        with st.chat_message(
            "user"
        ):

            st.markdown(
                prompt
            )

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        # Generate answer
        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Learnova is thinking..."
            ):

                answer, results = (
                    generate_rag_answer(
                        prompt,
                        difficulty
                    )
                )

            st.markdown(
                answer
            )

            source_data = []

            for document, metadata in results:

                source_data.append(
                    {
                        "text": document,
                        "chunk": metadata.get(
                            "chunk",
                            0
                        )
                    }
                )

            if source_data:

                with st.expander(
                    "🔎 View retrieved sources"
                ):

                    for source in source_data:

                        st.markdown(
                            f"""
                            **Chunk {source["chunk"] + 1}**

                            {source["text"]}
                            """
                        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": source_data
            }
        )