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


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Learnova",
    page_icon="🎓",
    layout="centered"
)

# ============================================================
# CUSTOM UI STYLING
# ============================================================

st.markdown("""
<style>

    /* Main application */
    .main {
        padding-top: 1rem;
    }

    /* Main title */
    h1 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
    }

    /* Section headings */
    h2, h3 {
        font-weight: 600 !important;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1rem;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border-radius: 12px;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 0.5rem;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        border-radius: 10px;
        padding: 10px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        padding-top: 1rem;
    }

    /* Dividers */
    hr {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

</style>
""", unsafe_allow_html=True)


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
    st.session_state.answers = []

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
    st.session_state.summary = None

if "quiz_version" not in st.session_state:
    st.session_state.quiz_version = 0


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🎓 Learnova")

    st.caption("Your AI-powered learning assistant")

    st.divider()

    st.subheader("✨ Features")

    st.write("📄 PDF-based Q&A")
    st.write("🔎 Semantic search")
    st.write("🧠 RAG-powered answers")
    st.write("📝 AI summaries")
    st.write("🎯 Automatic MCQs")
    st.write("📊 Quiz scoring")
    st.write("🎓 Personalized explanations")

    st.divider()

    st.caption(
        "Built with Python, Streamlit, Gemini API, "
        "RAG, Embeddings & ChromaDB"
    )

    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# GEMINI API SETUP
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:

    st.error(
        "⚠️ GEMINI_API_KEY is not configured. "
        "Please add it to your .env file or Streamlit Cloud Secrets."
    )

    st.stop()


client = genai.Client(api_key=api_key)


# ============================================================
# GEMINI GENERATION WITH RETRIES
# ============================================================

def generate_with_fallback(prompt):

    models = [
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite"
    ]

    last_error = None

    for model in models:

        for attempt in range(2):

            try:

                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )

                if response and response.text:

                    return response.text.strip()

                last_error = Exception("Empty Gemini response.")

            except Exception as e:

                last_error = e

                error_text = str(e).upper()

                if (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                    or "SERVICE UNAVAILABLE" in error_text
                ):

                    time.sleep(2 * (attempt + 1))

                    continue

                break

    return (
        "⚠️ Gemini is temporarily unavailable. "
        "Please try again in a moment."
    )


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ============================================================
# CHROMADB SETUP
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

    chunks = []

    start = 0

    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():

            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks


# ============================================================
# FILE HASH
# ============================================================

def calculate_file_hash(file_bytes):

    return hashlib.sha256(
        file_bytes
    ).hexdigest()


# ============================================================
# CLEAR CHROMADB
# ============================================================

def clear_collection():

    count = collection.count()

    if count > 0:

        existing_data = collection.get()

        existing_ids = existing_data.get(
            "ids",
            []
        )

        if existing_ids:

            collection.delete(
                ids=existing_ids
            )


# ============================================================
# GET ALL DOCUMENTS IN CORRECT CHUNK ORDER
# ============================================================

def get_all_documents():

    data = collection.get()

    documents = data.get(
        "documents",
        []
    )

    metadatas = data.get(
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
        key=lambda item: item[1].get(
            "chunk",
            0
        )
    )

    return combined


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def search_documents(
    query,
    top_k=3
):

    count = collection.count()

    if count == 0:

        return []


    query_embedding = embedding_model.encode(
        query
    ).tolist()


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
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]


    if not documents:

        return []


    return list(
        zip(
            documents,
            metadatas
        )
    )


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
            "I couldn't find the answer "
            "in the uploaded material."
        ), []


    context = "\n\n".join(

        document

        for document, metadata
        in results
    )


    prompt = f"""
You are Learnova, an AI learning assistant.

Answer the student's question using ONLY
the information provided in the context below.

Learning Level: {difficulty}

Adapt your explanation according to the student's level.

Beginner:
- Use very simple language.
- Explain basic terms.
- Give a simple example when useful.

Intermediate:
- Give a clear technical explanation.
- Assume basic knowledge of the topic.

Advanced:
- Give a detailed technical explanation.
- Include deeper concepts and technical terminology when relevant.

Important rules:
- Use ONLY the provided context.
- Do not use outside knowledge.
- Do not invent facts.
- If the answer is not present in the context, say:
  "I couldn't find the answer in the uploaded material."
- Answer clearly and directly.

CONTEXT:
{context}

QUESTION:
{question}
"""


    answer = generate_with_fallback(
        prompt
    )


    return answer, results


# ============================================================
# SUMMARY GENERATION
# ============================================================

def generate_summary():

    combined_documents = get_all_documents()


    if not combined_documents:

        return "No study material found."


    documents = [

        document

        for document, metadata
        in combined_documents
    ]


    # --------------------------------------------------------
    # Create partial summaries
    # --------------------------------------------------------

    batch_size = 5

    partial_summaries = []


    for i in range(
        0,
        len(documents),
        batch_size
    ):

        batch = documents[
            i:i + batch_size
        ]


        context = "\n\n".join(
            batch
        )


        prompt = f"""
You are Learnova, an AI learning assistant.

Summarize the following study material.

Rules:
- Use ONLY the provided study material.
- Identify the most important concepts.
- Keep important definitions.
- Keep important key points.
- Use headings and bullet points.
- Do not add outside information.
- Do not invent facts.

STUDY MATERIAL:

{context}
"""


        summary = generate_with_fallback(
            prompt
        )


        if summary.startswith(
            "⚠️ Gemini is temporarily unavailable"
        ):

            return summary


        partial_summaries.append(
            summary
        )


    # --------------------------------------------------------
    # Combine partial summaries
    # --------------------------------------------------------

    combined = "\n\n".join(
        partial_summaries
    )


    final_prompt = f"""
You are Learnova, an AI learning assistant.

Create ONE clear final study summary
from the partial summaries below.

Rules:
- Use ONLY the information provided.
- Remove unnecessary repetition.
- Keep important concepts.
- Keep important definitions.
- Keep important key points.
- Organize using headings and bullet points.
- Make it useful for exam revision.
- Do not add outside information.
- Do not invent facts.

PARTIAL SUMMARIES:

{combined}
"""


    final_summary = generate_with_fallback(
        final_prompt
    )


    return final_summary


# ============================================================
# SELECT REPRESENTATIVE CHUNKS FOR QUIZ
# ============================================================

def select_quiz_documents(
    documents,
    max_chunks=10
):

    if len(documents) <= max_chunks:

        return documents


    selected = []

    step = len(documents) / max_chunks


    for i in range(max_chunks):

        index = int(
            i * step
        )

        selected.append(
            documents[index]
        )


    return selected


# ============================================================
# EXTRACT JSON FROM GEMINI RESPONSE
# ============================================================

def extract_json(text):

    text = text.strip()


    # Remove markdown code blocks

    if text.startswith("```"):

        text = re.sub(
            r"```(?:json)?",
            "",
            text,
            flags=re.IGNORECASE
        )

        text = text.replace(
            "```",
            ""
        ).strip()


    # Try direct JSON

    try:

        return json.loads(text)

    except json.JSONDecodeError:

        pass


    # Try to find JSON array

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


    for question in quiz[:5]:

        if not isinstance(
            question,
            dict
        ):

            return False


        required_fields = [
            "question",
            "options",
            "answer",
            "explanation"
        ]


        for field in required_fields:

            if field not in question:

                return False


        options = question["options"]


        if not isinstance(
            options,
            list
        ):

            return False


        if len(options) != 4:

            return False


        if question["answer"] not in options:

            return False


    return True


# ============================================================
# GENERATE QUIZ
# ============================================================

def generate_quiz():

    combined_documents = get_all_documents()


    if not combined_documents:

        return []


    documents = [

        document

        for document, metadata
        in combined_documents
    ]


    # Use representative chunks from
    # the entire document

    selected_documents = select_quiz_documents(
        documents,
        max_chunks=10
    )


    context = "\n\n".join(
        selected_documents
    )


    prompt = f"""
You are Learnova, an AI learning assistant.

Create exactly 5 multiple-choice questions
based ONLY on the study material below.

Rules:

- Create exactly 5 questions.
- Each question must have exactly 4 options.
- There must be exactly one correct answer.
- The correct answer must exactly match one option.
- Do not use information outside the study material.
- Each question must include an explanation.
- Avoid duplicate questions.
- Cover different concepts from the provided material.
- Return ONLY valid JSON.
- Do not use markdown.
- Do not add any text before or after the JSON.

Use exactly this format:

[
  {{
    "question": "Question text",
    "options": [
      "Option 1",
      "Option 2",
      "Option 3",
      "Option 4"
    ],
    "answer": "Correct option",
    "explanation": "Explanation of the correct answer"
  }}
]

STUDY MATERIAL:

{context}
"""


    quiz_text = generate_with_fallback(
        prompt
    )


    if quiz_text.startswith(
        "⚠️ Gemini is temporarily unavailable"
    ):

        st.error(
            quiz_text
        )

        return []


    quiz = extract_json(
        quiz_text
    )


    if not validate_quiz(
        quiz
    ):

        st.error(
            "❌ Gemini returned invalid quiz data. "
            "Please try generating the quiz again."
        )

        return []


    return quiz[:5]


# ============================================================
# MAIN HEADER
# ============================================================

st.title("🎓 Learnova")

st.caption(
    "Your AI-powered personalized learning assistant"
)


# ============================================================
# LEARNING LEVEL
# ============================================================

st.subheader(
    "🎯 Choose Your Learning Level"
)


difficulty = st.selectbox(

    "How would you like Learnova to explain concepts?",

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
    "📚 Upload Study Material"
)


uploaded_file = st.file_uploader(

    "Upload your study material as a PDF",

    type=["pdf"],

    help="Upload lecture notes, textbooks, or study material."
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file:

    pdf_bytes = uploaded_file.getvalue()

    current_file_hash = calculate_file_hash(
        pdf_bytes
    )


    # Process only if the PDF is new/changed

    if (
        st.session_state.processed_file_hash
        != current_file_hash
    ):

        with st.spinner(
            "📖 Processing your PDF..."
        ):

            try:

                # ------------------------------------------------
                # Clear old document data
                # ------------------------------------------------

                clear_collection()


                # Clear previous results

                st.session_state.summary = None

                st.session_state.quiz = []

                st.session_state.quiz_submitted = False

                st.session_state.score = 0

                st.session_state.answers = []

                st.session_state.quiz_version += 1


                # ------------------------------------------------
                # Extract PDF text
                # ------------------------------------------------

                doc = fitz.open(
                    stream=pdf_bytes,
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


                if not text.strip():

                    st.error(
                        "❌ No text could be extracted "
                        "from this PDF."
                    )

                    st.stop()


                # ------------------------------------------------
                # Chunk text
                # ------------------------------------------------

                chunks = chunk_text(
                    text,
                    chunk_size=1000,
                    overlap=150
                )


                if not chunks:

                    st.error(
                        "❌ Could not create text chunks."
                    )

                    st.stop()


                # ------------------------------------------------
                # Create embeddings
                # ------------------------------------------------

                embeddings = embedding_model.encode(
                    chunks,
                    show_progress_bar=False
                )


                # ------------------------------------------------
                # Create unique IDs
                # ------------------------------------------------

                safe_file_name = re.sub(
                    r"[^a-zA-Z0-9_.-]",
                    "_",
                    uploaded_file.name
                )


                ids = [

                    f"{safe_file_name}_chunk_{i}"

                    for i in range(
                        len(chunks)
                    )
                ]


                # ------------------------------------------------
                # Metadata
                # ------------------------------------------------

                metadatas = [

                    {
                        "source": uploaded_file.name,
                        "chunk": i
                    }

                    for i in range(
                        len(chunks)
                    )
                ]


                # ------------------------------------------------
                # Store in ChromaDB
                # ------------------------------------------------

                collection.upsert(

                    ids=ids,

                    documents=chunks,

                    embeddings=embeddings.tolist(),

                    metadatas=metadatas
                )


                # ------------------------------------------------
                # Save session information
                # ------------------------------------------------

                st.session_state.processed_file_hash = (
                    current_file_hash
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


            except Exception as e:

                st.error(
                    f"❌ Error while processing PDF: {e}"
                )

                st.stop()


    # ========================================================
    # PDF STATUS
    # ========================================================

    st.success(
        f"✅ {uploaded_file.name} processed successfully!"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "📄 Pages",
            "Processed"
        )


    with col2:

        st.metric(
            "📚 Chunks",
            st.session_state.chunk_count
        )


    with col3:

        st.metric(
            "🧠 Embeddings",
            st.session_state.embedding_count
        )


    # ========================================================
    # EXTRACTED TEXT
    # ========================================================

    with st.expander(
        "📄 View Extracted Text"
    ):

        st.text_area(

            "PDF Content",

            st.session_state.pdf_text,

            height=300,

            label_visibility="collapsed"
        )


# ============================================================
# SUMMARY
# ============================================================

st.subheader(
    "📖 AI Study Summary"
)


if uploaded_file:

    if st.button(
        "✨ Summarize My Notes",
        use_container_width=True
    ):

        with st.spinner(
            "🧠 Creating your summary..."
        ):

            summary = generate_summary()

            st.session_state.summary = summary


else:

    st.info(
        "📚 Upload a PDF to generate an AI summary."
    )


if st.session_state.summary:

    st.markdown(
        "### 📝 Your Study Summary"
    )

    st.markdown(
        st.session_state.summary
    )


# ============================================================
# QUIZ
# ============================================================

st.subheader(
    "📝 AI Quiz"
)


if uploaded_file:

    if st.button(
        "🎯 Generate 5 MCQs",
        use_container_width=True
    ):

        with st.spinner(
            "🧠 Creating your quiz..."
        ):

            quiz = generate_quiz()


            if quiz:

                st.session_state.quiz = quiz

                st.session_state.quiz_submitted = False

                st.session_state.score = 0

                st.session_state.answers = []

                st.session_state.quiz_version += 1


else:

    st.info(
        "📚 Upload a PDF to generate a quiz."
    )


# ============================================================
# DISPLAY QUIZ
# ============================================================

if st.session_state.quiz:

    st.write(
        "### 🎓 Test Your Knowledge"
    )


    answers = []


    quiz = st.session_state.quiz


    for i, question in enumerate(
        quiz
    ):

        st.markdown(
            f"**{i + 1}. {question['question']}**"
        )


        selected = st.radio(

            "Choose an answer:",

            question["options"],

            key=(
                f"quiz_{st.session_state.quiz_version}"
                f"_question_{i}"
            ),

            index=None
        )


        answers.append(
            selected
        )


        if i < len(quiz) - 1:

            st.divider()


    if st.button(
        "✅ Submit Quiz",
        use_container_width=True
    ):

        # Make sure every question has an answer

        if any(
            answer is None
            for answer in answers
        ):

            st.warning(
                "⚠️ Please answer all 5 questions "
                "before submitting."
            )

        else:

            score = 0


            for i, question in enumerate(
                quiz
            ):

                if (
                    answers[i]
                    == question["answer"]
                ):

                    score += 1


            st.session_state.score = score

            st.session_state.answers = answers

            st.session_state.quiz_submitted = True


# ============================================================
# QUIZ RESULTS
# ============================================================

if st.session_state.quiz_submitted:

    score = st.session_state.score

    total = len(
        st.session_state.quiz
    )


    percentage = int(
        (score / total) * 100
    )


    if percentage == 100:

        st.success(
            f"🏆 Excellent! You scored {score}/{total} ({percentage}%)"
        )

    elif percentage >= 60:

        st.success(
            f"🎉 Good job! You scored {score}/{total} ({percentage}%)"
        )

    else:

        st.warning(
            f"📚 Keep practicing! You scored {score}/{total} ({percentage}%)"
        )


    st.subheader(
        "📖 Answer Explanations"
    )


    for i, question in enumerate(
        st.session_state.quiz
    ):

        user_answer = (
            st.session_state.answers[i]
        )

        correct_answer = (
            question["answer"]
        )


        if user_answer == correct_answer:

            st.success(
                f"Question {i + 1}: Correct ✅"
            )

        else:

            st.error(
                f"Question {i + 1}: Incorrect ❌"
            )

            st.write(
                f"Your answer: **{user_answer}**"
            )

            st.write(
                f"Correct answer: **{correct_answer}**"
            )


        st.info(
            question.get(
                "explanation",
                "No explanation provided."
            )
        )


# ============================================================
# RAG CHAT
# ============================================================

st.subheader(
    "💬 Ask Learnova"
)


if not uploaded_file:

    st.info(
        "📚 Upload a PDF first, then ask questions "
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


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Ask something about your study material..."
)


if prompt:

    # --------------------------------------------------------
    # User message
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Assistant response
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        if collection.count() == 0:

            answer = (
                "📚 Please upload a PDF first "
                "so I can answer using your study material."
            )

            st.markdown(
                answer
            )

        else:

            with st.spinner(
                "🧠 Learnova is thinking..."
            ):

                answer, sources = (
                    generate_rag_answer(
                        prompt,
                        difficulty
                    )
                )


            st.markdown(
                answer
            )


            # ------------------------------------------------
            # Show retrieved sources
            # ------------------------------------------------

            if sources:

                with st.expander(
                    "🔎 View retrieved study material"
                ):

                    for i, (
                        document,
                        metadata
                    ) in enumerate(
                        sources
                    ):

                        chunk_number = (
                            metadata.get(
                                "chunk",
                                i
                            ) + 1
                        )


                        st.markdown(
                            f"**Source chunk {chunk_number}**"
                        )


                        st.write(
                            document
                        )


                        if i < len(sources) - 1:

                            st.divider()


    # --------------------------------------------------------
    # Save assistant response
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )