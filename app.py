from sentence_transformers import SentenceTransformer
import streamlit as st
from google import genai
import os
from dotenv import load_dotenv
import fitz
import chromadb
import json
import time


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
# INITIALIZE SESSION STATE
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

if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🎓 Learnova")

    st.write("### Your AI Learning Assistant")

    st.divider()

    st.write("📚 **Features**")
    st.write("• PDF-based Q&A")
    st.write("• RAG-powered answers")
    st.write("• AI summaries")
    st.write("• Automatic MCQs")
    st.write("• Quiz scoring")
    st.write("• Personalized explanations")

    st.divider()

    st.caption("Built with Python, Gemini, RAG & ChromaDB")

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()



# ============================================================
# GEMINI CLIENT
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY is not set in your .env file.")
    st.stop()

client = genai.Client(api_key=api_key)




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
                return response.text

            except Exception as e:
                last_error = e

                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    time.sleep(2 * (attempt + 1))
                    continue

                break

    return (
        "⚠️ Gemini is temporarily unavailable. "
        "Please try again in a moment."
    )


# ============================================================
# EMBEDDING MODEL
# ============================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# CHROMADB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path="./vectorstore"
)

collection = chroma_client.get_or_create_collection(
    name="learnova_documents"
)


# ============================================================
# TEXT CHUNKING
# ============================================================

def chunk_text(text, chunk_size=1000):

    chunks = []

    for i in range(0, len(text), chunk_size):

        chunk = text[i:i + chunk_size]

        if chunk.strip():
            chunks.append(chunk)

    return chunks


# ============================================================
# SEARCH DOCUMENTS
# ============================================================

def search_documents(query, top_k=3):

    # Make sure there are documents
    count = collection.count()

    if count == 0:
        return []

    # Create embedding for the question
    query_embedding = embedding_model.encode(
        query
    ).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count)
    )

    if not results.get("documents"):
        return []

    if not results["documents"][0]:
        return []

    return results["documents"][0]


# ============================================================
# GENERATE RAG ANSWER
# ============================================================

def generate_rag_answer(question, difficulty):

    relevant_chunks = search_documents(question)

    if not relevant_chunks:

        return (
            "I couldn't find the answer in the "
            "uploaded material."
        )

    # Combine relevant chunks
    context = "\n\n".join(
        relevant_chunks
    )

    prompt = f"""
You are Learnova, an AI learning assistant.

Answer the student's question using ONLY
the information provided in the context below.

Learning Level: {difficulty}

Adapt your explanation according to the student's level:

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


If the answer cannot be found in the context,
say:

"I couldn't find the answer in the uploaded material."

Explain the answer clearly and simply.

CONTEXT:
{context}

QUESTION:
{question}
"""

    return generate_with_fallback(prompt)

def generate_summary():

    documents = collection.get()["documents"]

    if not documents:
        return "No study material found."

    batch_size = 5
    partial_summaries = []

    # Summarize each batch
    for i in range(0, len(documents), batch_size):

        batch = documents[i:i + batch_size]

        context = "\n\n".join(batch)

        prompt = f"""
You are Learnova, an AI learning assistant.

Summarize the following study material.

Rules:
- Use ONLY the provided study material.
- Identify the most important concepts.
- Keep important definitions and key points.
- Use headings and bullet points.
- Do not add information that is not present in the material.

STUDY MATERIAL:
{context}
"""

        summary = generate_with_fallback(prompt)

        if summary.startswith("⚠️ Gemini is temporarily unavailable"):
            return summary

        partial_summaries.append(summary)

    # Combine all partial summaries
    combined = "\n\n".join(partial_summaries)

    final_prompt = f"""
You are Learnova, an AI learning assistant.

Create one clear final study summary from the partial summaries below.

Rules:
- Use ONLY the information provided.
- Remove unnecessary repetition.
- Keep important concepts, definitions and key points.
- Organize the answer with headings and bullet points.
- Make it useful for exam revision.
- Do not add outside information.

PARTIAL SUMMARIES:
{combined}
"""

    final_summary = generate_with_fallback(final_prompt)

    return final_summary
# ============================================================
# GENERATE QUIZ
# ============================================================

def generate_quiz():

    data = collection.get()

    documents = data.get("documents", [])

    if not documents:
        return []

    # Use first 10 chunks
    context = "\n\n".join(documents[:10])

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
- Return ONLY valid JSON.
- Do not use markdown code blocks.

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

    quiz_text = generate_with_fallback(prompt)

    # Check if Gemini failed
    if quiz_text.startswith("⚠️ Gemini is temporarily unavailable"):
        st.error(quiz_text)
        return []

    # Remove markdown code blocks if Gemini adds them
    if quiz_text.startswith("```"):
        quiz_text = quiz_text.replace(
            "```json", ""
        ).replace(
            "```", ""
        ).strip()

    try:

        quiz = json.loads(quiz_text)

        if not isinstance(quiz, list):
            return []

        return quiz[:5]

    except json.JSONDecodeError:

        st.error(
            "❌ Gemini returned invalid quiz data."
        )

        return []
# ============================================================
# TITLE
# ============================================================

st.title("🎓 Learnova")

st.caption(
    "Your AI-powered learning assistant"
)

st.subheader("🎯 Choose Your Learning Level")

difficulty = st.selectbox(
    "How would you like Learnova to explain concepts?",
    ["Beginner", "Intermediate", "Advanced"]
)



# ============================================================
# PDF UPLOAD
# ============================================================

st.subheader(
    "📚 Upload Study Material"
)

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


# ============================================================
# PROCESS PDF
# ============================================================

# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file:

    # Check if this PDF is already processed
    if st.session_state.processed_file != uploaded_file.name:

        with st.spinner("Processing your PDF..."):

            # Open PDF
            pdf_bytes = uploaded_file.read()

            doc = fitz.open(
                stream=pdf_bytes,
                filetype="pdf"
            )

            # Extract text
            text = ""

            for page in doc:
                text += page.get_text()

            doc.close()

            # Check extracted text
            if not text.strip():

                st.error(
                    "❌ No text could be extracted from this PDF."
                )

            else:

                # Create chunks
                chunks = chunk_text(text)

                # Create embeddings
                embeddings = embedding_model.encode(chunks)

                # Create unique file ID
                file_id = uploaded_file.name.replace(
                    " ",
                    "_"
                )

                ids = [
                    f"{file_id}_chunk_{i}"
                    for i in range(len(chunks))
                ]

                # Store in ChromaDB
                collection.upsert(
                    ids=ids,
                    documents=chunks,
                    embeddings=embeddings.tolist()
                )

                # Save PDF information in session state
                st.session_state.processed_file = uploaded_file.name
                st.session_state.pdf_text = text
                st.session_state.chunk_count = len(chunks)
                st.session_state.embedding_count = len(embeddings)

                # Clear old summary and quiz
                st.session_state.pop("summary", None)
                st.session_state.quiz = []
                st.session_state.quiz_submitted = False
                st.session_state.score = 0
                st.session_state.answers = []

    # ========================================================
    # DISPLAY PDF INFORMATION
    # ========================================================

    if "pdf_text" in st.session_state:

        st.success("PDF uploaded successfully! ✅")

        st.write("📄 Extracted text:")

        st.text_area(
            "PDF Content",
            st.session_state.pdf_text,
            height=300
        )

        st.write(
            f"📚 Created {st.session_state.chunk_count} chunks"
        )

        st.write(
            f"🧠 Created {st.session_state.embedding_count} embeddings"
        )

        st.success("📦 Chunks stored in ChromaDB! ✅")
# ============================================================
# AI SUMMARY
# ============================================================

st.subheader("📖 AI Study Summary")

if uploaded_file:

    if st.button("✨ Summarize My Notes"):

        with st.spinner("Creating your summary..."):

            summary = generate_summary()

            st.session_state.summary = summary


# ============================================================
# DISPLAY SUMMARY
# ============================================================

if "summary" in st.session_state:

    st.markdown("### 📝 Summary")

    st.markdown(
        st.session_state.summary
    )


# ============================================================
# QUIZ
# ============================================================

st.subheader("📝 AI Quiz")


if uploaded_file:

    if st.button(
        "🎯 Generate 5 MCQs"
    ):

        with st.spinner(
            "Creating your quiz..."
        ):

            quiz = generate_quiz()

            if quiz:

                st.session_state.quiz = quiz

                st.session_state.quiz_submitted = False

                st.session_state.score = 0

                st.session_state.answers = []


# ============================================================
# DISPLAY QUIZ
# ============================================================

if st.session_state.quiz:

    st.write(
        "### 📚 Test Your Knowledge"
    )

    answers = []

    quiz = st.session_state.quiz

    for i, question in enumerate(quiz):

        st.write(
            f"**{i + 1}. "
            f"{question['question']}**"
        )

        selected = st.radio(
            "Choose an answer:",
            question["options"],
            key=f"question_{i}"
        )

        answers.append(selected)


    # --------------------------------------------------------
    # SUBMIT QUIZ
    # --------------------------------------------------------

    if st.button(
        "✅ Submit Quiz"
    ):

        score = 0

        for i, question in enumerate(quiz):

            if answers[i] == question["answer"]:
                score += 1

        st.session_state.score = score

        st.session_state.answers = answers

        st.session_state.quiz_submitted = True


# ============================================================
# SHOW SCORE AND EXPLANATIONS
# ============================================================

if st.session_state.quiz_submitted:

    score = st.session_state.score

    st.success(
        f"🎉 Your score: {score}/{len(st.session_state.quiz)}"
    )

    st.subheader(
        "📖 Explanations"
    )

    for i, question in enumerate(
        st.session_state.quiz
    ):

        user_answer = (
            st.session_state.answers[i]
        )

        correct_answer = question["answer"]

        if user_answer == correct_answer:

            st.success(
                f"Question {i + 1}: Correct ✅"
            )

        else:

            st.error(
                f"Question {i + 1}: Incorrect ❌"
            )

            st.write(
                f"Correct answer: "
                f"**{correct_answer}**"
            )

        # Explanation
        explanation = question.get(
            "explanation",
            "No explanation provided."
        )

        st.info(explanation)


# ============================================================
# CHAT HISTORY
# ============================================================

st.subheader(
    "💬 Ask Learnova"
)


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
    "Ask Learnova anything..."
)


# ============================================================
# GENERATE AI RESPONSE
# ============================================================

if prompt:

    # Display user message
    with st.chat_message("user"):

        st.markdown(prompt)

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Generate RAG answer
    with st.chat_message("assistant"):

        with st.spinner(
            "Learnova is thinking..."
        ):

            answer = generate_rag_answer(prompt, difficulty)

            st.markdown(answer)

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

