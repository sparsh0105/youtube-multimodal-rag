import streamlit as st
import os
from pathlib import Path
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from core.llm import get_llm
from core.vectorstore import get_vectorstore, get_retriever
from core.rag import answer_question
from core.memory import get_effective_question
from core.youtube import extract_video_id, download_audio, download_video
from core.audio import transcribe_and_chunk
from core.video import extract_frames
from core.captions import extract_visual_captions
from core.chunking import (
    load_fixed_chunks,
    load_visual_captions,
    build_multimodal_chunks
)

from utils.parallel import run_parallel
from utils.gpu import print_gpu_info
from dotenv import load_dotenv
load_dotenv()
# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="YouTube Multimodal RAG", layout="wide")
st.title("🎥 YouTube Multimodal RAG Chatbot")

# -------------------------------------------------
# Debug / GPU info (optional)
# -------------------------------------------------
with st.expander("System Info"):
    print_gpu_info()
    st.write("OPENAI_API_KEY set:", bool(os.environ.get("OPENAI_API_KEY")))

# -------------------------------------------------
# Session state init
# -------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "video_id" not in st.session_state:
    st.session_state.video_id = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# -------------------------------------------------
# Video input
# -------------------------------------------------
video_url = st.text_input(
    "Enter YouTube URL",
    placeholder="https://youtu.be/xxxxxxxx"
)

# -------------------------------------------------
# Process video (runs once per video)
# -------------------------------------------------
if st.button("Process Video") and video_url:

    st.session_state.chat_history = []

    try:
        # Extract video ID using proper function
        video_id = extract_video_id(video_url)
        st.session_state.video_id = video_id
    except ValueError as e:
        st.error(f"Invalid YouTube URL: {e}")
        st.stop()

    with st.spinner("Processing video (cached if already done)..."):
        try:
            # Fast path: Check if vector store already exists
            persist_dir = Path(f"cache/chroma/{video_id}")
            if persist_dir.exists() and list(persist_dir.iterdir()):
                # Vector store exists, load it directly (skip all processing)
                embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vector_store = Chroma(
                    persist_directory=str(persist_dir),
                    embedding_function=embedding
                )
                retriever = get_retriever(vector_store)
                st.session_state.retriever = retriever
            else:
                # Need to process: Step 1: Download audio and video in parallel (only if needed)
                run_parallel({
                    "audio": lambda: download_audio(video_id, video_url),
                    "video": lambda: download_video(video_id, video_url),
                })

                # Step 2: Transcribe audio and extract frames in parallel (only if needed)
                run_parallel({
                    "transcript": lambda: transcribe_and_chunk(video_id),
                    "frames": lambda: extract_frames(video_id),
                })

                # Step 3: Extract visual captions (only if needed, depends on frames)
                extract_visual_captions(video_id)

                # Step 4: Load cached artifacts and build multimodal chunks
                fixed_chunks = load_fixed_chunks(video_id)
                visual_captions = load_visual_captions(video_id)

                multimodal_chunks = build_multimodal_chunks(
                    video_id,
                    fixed_chunks,
                    visual_captions
                )

                # Step 5: Create vector store and retriever
                vector_store = get_vectorstore(video_id, multimodal_chunks)
                retriever = get_retriever(vector_store)

                st.session_state.retriever = retriever

        except FileNotFoundError as e:
            st.error(f"Processing error: {e}")
            st.info("Make sure all processing steps completed successfully.")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

    st.success("Video ready! Ask questions below 👇")

# -------------------------------------------------
# Chat interface
# -------------------------------------------------
if st.session_state.retriever:

    llm = get_llm()

    user_question = st.chat_input("Ask something about the video")

    if user_question:

        question = get_effective_question(
            llm=llm,
            chat_history=st.session_state.chat_history,
            user_question=user_question,
            n_turns=3
        )

        answer = answer_question(
            llm=llm,
            retriever=st.session_state.retriever,
            question=question
        )

        st.session_state.chat_history.append(("user", user_question))
        st.session_state.chat_history.append(("assistant", answer))

    # Render chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)

else:
    st.info("Enter a YouTube URL and click **Process Video**")
