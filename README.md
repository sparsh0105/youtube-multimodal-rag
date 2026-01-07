# 🎥 YouTube Multimodal RAG

A GPU-accelerated multimodal Retrieval-Augmented Generation (RAG) system that allows users to chat with YouTube videos using both audio transcripts and visual context from video frames.

This project extracts, processes, and caches multimodal information from YouTube videos and enables conversational question-answering with timestamped evidence.

## 🏗️ Architecture Overview

```
YouTube URL
   ↓
Extract Video ID
   ↓
Audio Download ──┐
Video Download ──┘ (parallel)
   ↓
Whisper Transcription (10s windows)
BLIP Frame Captioning (10s interval)
   ↓
Multimodal Chunking
   ↓
Chroma Vector Database (per video)
   ↓
Conversational RAG (LangChain + OpenAI)
   ↓
Streamlit Chat UI
```

## ✨ Features

### 🔎 Multimodal RAG

- Audio transcripts (Whisper)
- Visual captions from video frames (BLIP)
- Combined multimodal chunks for better grounding

### 💬 Conversational Chatbot

- Context-aware follow-up questions
- LLM-based question rewriting for memory handling

### ⚡ GPU Accelerated

- Uses CUDA when available (Whisper + BLIP)
- Falls back to CPU automatically

### 🧠 Efficient Caching

- Avoids recomputation for the same video
- Separate cache per video (audio, frames, captions, embeddings)

### 🚀 Parallel Processing

- Audio & video downloads in parallel
- Heavy preprocessing optimized for speed

### 🧩 Clean, Modular Architecture

- Core logic separated from orchestration
- Easy to extend or deploy


## Use the Application

1. Paste a YouTube URL
2. Wait for processing (first time only, then cached)
3. Start asking questions about the video!

## 🎯 How It Works

1. **Download**: Downloads audio and video in parallel from YouTube
2. **Transcription**: Uses Whisper to transcribe audio in 10-second windows
3. **Captioning**: Extracts frames every 10 seconds and generates captions with BLIP
4. **Chunking**: Combines audio transcripts and visual captions into multimodal chunks
5. **Indexing**: Stores chunks in Chroma vector database with embeddings
6. **Retrieval**: Finds relevant chunks based on user question
7. **Generation**: Uses LLM to generate answer from retrieved context

## 🔍 Caching System

The system caches:
- ✅ Downloaded audio files
- ✅ Downloaded video files
- ✅ Extracted frames
- ✅ Whisper transcriptions
- ✅ BLIP captions
- ✅ Vector database embeddings

This means subsequent queries on the same video are **instant**!
