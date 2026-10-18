<div align="center">

# YouTube RAG System

A Retrieval-Augmented Generation (RAG) system for YouTube channel content. Index any YouTube channel and chat with its videos using AI-powered semantic search.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YouTube Data API](https://img.shields.io/badge/YouTube-Data%20API%20v3-FF0000.svg)](https://developers.google.com/youtube/v3)
[![yt-dlp](https://img.shields.io/badge/yt--dlp-transcript%20fetching-red.svg)](https://github.com/yt-dlp/yt-dlp)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063.svg)](https://docs.pydantic.dev/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-412991.svg)](https://openai.com/)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude-191919.svg)](https://www.anthropic.com/)
[![LangChain](https://img.shields.io/badge/LangChain-text%20splitters-1C3C3C.svg)](https://www.langchain.com/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-green.svg)](https://www.trychroma.com/)

</div>

<div align="center">

![Demo](./public/assets/demo_video.gif)

  <p><em>See the YouTube RAG system in action: indexing channels and chatting with video content.</em></p>

</div>

## 📋 Table of Contents

- [📖 Overview](#-overview)
- [✨ Features](#-features)
- [🚀 Installation](#-installation)
- [💻 Usage](#-usage)
- [⚙️ Configuration](#️-configuration)
- [📁 Project Structure](#-project-structure)
- [🏗️ Architecture](#️-architecture)
- [🔧 Troubleshooting](#-troubleshooting)

## 📖 Overview

YouTube RAG System enables you to build a searchable knowledge base from any YouTube channel. It fetches video transcripts, processes them into semantic chunks, stores them in a vector database, and provides an interactive chat interface where you can ask questions about the channel's content.

### 💡 Inspiration

This project was inspired by my interest in understanding how creators within the same YouTube niche think and differentiate themselves. The most valuable insights live in what creators say in their videos, not just in titles, descriptions, or metadata.

However, current LLMs like OpenAI and Claude can’t reliably access or reason over spoken video content, making deep niche analysis difficult and incomplete. This limitation motivated the project, which is to enable meaningful exploration of creator niches through their actual narratives, rather than surface-level signals.

**How it works:**

1. Provide a YouTube channel handle (e.g., `@mkbhd`)
2. System fetches all videos and their public transcripts using YouTube Data API v3 and youtube-transcript-api (with a cookie-free yt-dlp fallback)
3. Transcripts are chunked and embedded into a vector database (ChromaDB)
4. Ask questions in natural language via a Gradio web interface
5. Get AI-generated answers based on relevant video segments with source citations

**Use cases:**

- Research and summarize educational channels
- Extract insights from tech review channels
- Build a searchable knowledge base for tutorial channels
- Analyze content patterns across a creator's videos

## ✨ Features

- **YouTube Integration**: Fetch channel videos with YouTube Data API v3 and public captions with youtube-transcript-api, with a cookie-free yt-dlp fallback
- **Vector Database**: Semantic search powered by ChromaDB
- **Interactive UI**: Clean chat interface built with Gradio
- **Multiple LLM Providers**: Support for OpenAI and Anthropic
- **Flexible Embeddings**: Choose between OpenAI embeddings or local sentence transformers (free)
- **Smart Chunking**: Recursive text splitting with configurable overlap for context preservation
- **Auto-skip**: Automatically skips videos without available transcripts
- **Source Citations**: Every answer includes links to the source videos
- **VSCode Integration**: Pre-configured debug settings included
- **CLI Tools**: Scripts for indexing, inspection, and testing

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- [YouTube Data API v3 key](https://console.cloud.google.com/apis/credentials) (required)
- OpenAI API key (if using OpenAI for embeddings/LLM) **OR** Anthropic API key (if using Claude)

### Quick Start (Recommended)

Using the included Makefile for one-command setup:

```bash
# 1. Clone the repository
git clone git@github.com:Akirachang/youtube-RAG.git
cd youtube-rag

# 2. Setup: creates venv, installs deps, creates .env
make setup

# 3. Activate virtual environment
source venv/bin/activate

# 4. Edit .env with your API keys
nano .env
# Add your YOUTUBE_API_KEY and LLM provider keys

# 5. Index a YouTube channel
make index CHANNEL=@veritasium MAX_VIDEOS=50

# 6. Run the Gradio app
make run
```

Open your browser to `http://localhost:7860` and start chatting!

### Manual Installation

If you prefer manual setup:

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -e .

# 3. Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# 4. Run the application
python -m src.app
```

### Available Make Commands

```bash
make help          # Show all available commands
make setup         # Complete setup (venv + deps + .env)
make install       # Install/update dependencies
make run           # Start Gradio web interface
make index         # Index a channel (usage: make index CHANNEL=@name MAX_VIDEOS=50)
make inspect       # Inspect indexed videos and chunks
make clean         # Remove temporary files
make clean-data    # Remove vector database data only
make clean-all     # Remove everything including venv
make test          # Run tests
make format        # Format code with black
make lint          # Run linting checks
```

## 💻 Usage

### Using the Web Interface

1. Start the Gradio app:

   ```bash
   make run
   ```

2. Open your browser to `http://localhost:7860`

3. **Index a channel:**

   - Go to the "Index Channel" tab
   - Enter a channel handle (e.g., `@veritasium`, `@mkbhd`)
   - Set max videos to index (default: 50)
   - Click "Index Channel" and wait for completion

4. **Chat with the content:**
   - Go to the "Chat" tab
   - Ask questions about the channel's videos
   - Get AI-generated answers with source citations

### Using the CLI

Index a channel from the command line:

```bash
# Index with default settings (1 video from @TheDailyShow)
make index

# Index a specific channel
make index CHANNEL=@veritasium

# Index with more videos
make index CHANNEL=@mkbhd MAX_VIDEOS=100

# Or run the script directly
python scripts/test_indexer.py @veritasium --max-videos 50

# Clear and re-index
python scripts/test_indexer.py @veritasium --clear --max-videos 50
```

Inspect indexed data:

```bash
# View indexed channels and videos
make inspect

# Or use the script directly
python scripts/inspect_index.py
```

Test chat functionality:

```bash
# Interactive chat testing
python scripts/test_chat.py
```

### Programmatic Usage

Use the services in your own Python code:

```python
from src.services.indexing import IndexingService
from src.services.chat import ChatService

# Index a channel
indexing = IndexingService()
stats = indexing.index_channel("@veritasium", max_videos=50)
print(f"Indexed {stats['videos_indexed']} videos")
print(f"Created {stats['total_chunks']} chunks")

# Query the system
chat = ChatService()
result = chat.ask("What topics does this channel cover?")
print(result["answer"])

# Access source videos
for source in result["sources"]:
    print(f"- {source['video_title']}")
    print(f"  https://www.youtube.com/watch?v={source['video_id']}")
```

## ⚙️ Configuration

The system is configured via environment variables in the `.env` file.

### Required Settings

```bash
# YouTube API (Required)
YOUTUBE_API_KEY=your_youtube_api_key_here
```

### LLM Provider Settings

Choose either OpenAI or Anthropic:

```bash
# Option 1: OpenAI (GPT-4)
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=your_openai_key_here

# Option 2: Anthropic (Claude)
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_API_KEY=your_anthropic_key_here
```

#### Google Gemini

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash
SHOULD_LOCAL_EMBED=true
```

Gemini uses the official `google-genai` Python SDK. `SHOULD_LOCAL_EMBED=true`
keeps embeddings local, so this configuration does not require OpenAI credits.

### Embedding Settings

Choose between local embeddings (free, runs on CPU) or OpenAI embeddings (paid, higher quality):

```bash
# Option 1: Local sentence transformers (FREE)
SHOULD_LOCAL_EMBED=true
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Option 2: OpenAI embeddings (requires API key)
SHOULD_LOCAL_EMBED=false
OPENAI_API_KEY=your_openai_key_here
```

**Important:** If you switch between local and OpenAI embeddings, run `make clean-data` to clear the vector database, as they use different dimensions (384 vs 1536).

### Optional Settings

```bash
# Vector Database
CHROMA_DB_PATH=./data/chroma_db

# Text Chunking
CHUNK_SIZE=1000          # Size of each text chunk
CHUNK_OVERLAP=200        # Overlap between chunks for context

# Retrieval
RETRIEVAL_K=5            # Number of similar chunks to retrieve

# Logging
LOG_LEVEL=INFO           # DEBUG, INFO, WARNING, ERROR
```

## 📁 Project Structure

```
youtube-rag/
├── src/
│   ├── app.py                    # Gradio web interface entry point
│   │
│   ├── youtube/                  # YouTube API integration
│   │   ├── base.py              # Abstract base class
│   │   ├── client.py            # Main YouTube API client
│   │   ├── channel.py           # Channel lookup (handle → ID)
│   │   ├── videos.py            # Fetch channel videos
│   │   └── transcripts.py       # Fetch video transcripts (yt-dlp)
│   │
│   ├── rag/                      # RAG pipeline components
│   │   ├── base.py              # Abstract base classes
│   │   ├── chunker.py           # Text chunking strategies
│   │   ├── embeddings.py        # OpenAI & local embedders
│   │   ├── vectorstore.py       # ChromaDB integration
│   │   ├── retriever.py         # Semantic document retrieval
│   │   └── generator.py         # LLM response generation
│   │
│   ├── core/                     # Core utilities
│   │   ├── config.py            # Pydantic settings management
│   │   ├── exceptions.py        # Custom exceptions
│   │   └── logging.py           # Logging configuration
│   │
│   └── services/                 # Business logic layer
│       ├── base.py              # Base service class
│       ├── indexing.py          # Channel indexing orchestration
│       └── chat.py              # Chat/query service
│
├── scripts/
│   ├── test_indexer.py          # Test script for indexing
│   ├── inspect_index.py         # View indexed data
│   ├── test_chat.py             # Test chat functionality
│   └── test_ytdlp_captions.py   # Test yt-dlp caption extraction
│
├── data/
│   ├── chroma_db/               # Vector database storage
│   └── cache/                   # Cache directory
│
├── tests/                        # Test suite
│   ├── unit/                    # Unit tests
│   └── integration/             # Integration tests
│
├── .vscode/                      # VSCode debug configurations
├── Makefile                      # Development automation
├── pyproject.toml               # Project dependencies
├── .env.example                 # Environment template
└── README.md                    # This file
```

## 🏗️ Architecture

The system follows a modular RAG (Retrieval-Augmented Generation) architecture:

### Data Flow

```
1. Indexing Pipeline:
   YouTube Channel → Fetch Videos → Extract Transcripts → Chunk Text →
   Generate Embeddings → Store in Vector DB

2. Query Pipeline:
   User Question → Generate Query Embedding → Retrieve Similar Chunks →
   Extract Context → Generate Answer with LLM → Return with Sources
```

### Component Breakdown

#### 1. YouTube Integration (`src/youtube/`)

- **ChannelFetcher**: Converts channel handles (`@veritasium`) to channel IDs
- **VideosFetcher**: Retrieves all videos from a channel using YouTube Data API v3
- **TranscriptFetcher**: Extracts public video transcripts using youtube-transcript-api, with a cookie-free yt-dlp fallback

##### Authorized captions for videos you manage

The YouTube Data API's normal API key can discover channel and video metadata,
but it cannot download caption text. The optional official Captions API provider
uses an OAuth token file only for videos the authenticated YouTube user can edit.
Set `YOUTUBE_OAUTH_TOKEN_FILE` to a private OAuth authorized-user token JSON file
with the `youtube.force-ssl` scope. It is attempted before public providers.

This does **not** grant access to captions on arbitrary public channels: Google
requires OAuth for `captions.list`, and `captions.download` requires permission to
edit the video. Keep the token outside source control. If an owned video has no
accessible caption track, or public automated retrieval is blocked, use the
existing **Import Transcript** tab with text you are authorized to use.

Manual authorized-provider test: create an OAuth authorized-user token with the
`https://www.googleapis.com/auth/youtube.force-ssl` scope for a YouTube account
that can edit a video with captions, keep the JSON file private, then set
`YOUTUBE_OAUTH_TOKEN_FILE=C:/private/youtube-oauth-token.json` in `.env` and run:

```powershell
python scripts\test_indexer.py "@your-channel-handle" --max-videos 1
```

The official provider is expected to work only for videos that account can edit;
it is not a way to retrieve caption text from arbitrary public channels.

#### 2. RAG Pipeline (`src/rag/`)

- **Chunker**: Splits long transcripts into overlapping chunks for better context

  - Uses `RecursiveCharacterTextSplitter` from LangChain
  - Configurable chunk size and overlap

- **Embedder**: Converts text chunks into vector representations

  - **LocalEmbedder**: Uses sentence-transformers (384 dimensions, free)
  - **OpenAIEmbedder**: Uses OpenAI's embedding API (1536 dimensions, paid)

- **VectorStore**: Manages the ChromaDB vector database

  - Stores embeddings with metadata (video title, channel, timestamps)
  - Supports similarity search and filtering

- **Retriever**: Finds the most relevant chunks for a given query

  - Embeds the user's question
  - Performs semantic similarity search
  - Returns top-k most relevant chunks

- **Generator**: Creates natural language answers using LLMs
  - **OpenAIGenerator**: Uses GPT-4 or GPT-3.5
  - **AnthropicGenerator**: Uses Claude models
  - Takes retrieved context and generates coherent responses

#### 3. Services Layer (`src/services/`)

- **IndexingService**: Orchestrates the entire indexing pipeline

  - Fetches channel data
  - Processes transcripts
  - Stores in vector database
  - Returns statistics

- **ChatService**: Handles user queries
  - Retrieves relevant context
  - Generates answers with citations
  - Returns formatted responses

#### 4. Configuration (`src/core/`)

- **Settings**: Pydantic-based configuration with validation
  - Loads from `.env` file
  - Type checking and defaults
  - Singleton pattern with `@lru_cache`

### Design Principles

- **Modularity**: Each component has a single responsibility
- **Extensibility**: Abstract base classes allow easy addition of new providers
- **Type Safety**: Full type hints throughout the codebase
- **Error Handling**: Graceful handling of missing transcripts and API errors
- **Caching**: Settings cached for performance

## 🔧 Troubleshooting

### Channel Not Found Error

Make sure you're using the correct format:

- ✅ Correct: `@mkbhd` or `mkbhd`
- ❌ Wrong: `https://www.youtube.com/@mkbhd` (don't include full URL)

### No Transcripts Available

Some channels disable transcripts or use languages other than English. The system automatically skips these videos. Check the indexing stats:

```
Videos Indexed: 45   ← Successfully indexed
Videos Skipped: 5    ← No transcripts available
```

### Config Validation Error

Make sure your `.env` file has **no inline comments**:

```bash
# ❌ Wrong - inline comments not supported
LLM_PROVIDER=openai  # this is a comment

# ✅ Correct - comments on separate lines
# Choose your LLM provider
LLM_PROVIDER=openai
```

### Embedding Dimension Mismatch

Error: `Collection expecting embedding with dimension of 384, got 1536`

This happens when you switch between local (384-dim) and OpenAI (1536-dim) embeddings. Fix:

```bash
# Clear the vector database
make clean-data

# Re-index with consistent settings
make index CHANNEL=@yourchannel MAX_VIDEOS=50
```

### Import Errors in VSCode

Make sure `PYTHONPATH` is set. The `.vscode/launch.json` includes this automatically, but if running manually:

```bash
export PYTHONPATH="${PWD}"
python scripts/test_indexer.py
```

### API Quota Exceeded

- **YouTube API**: Free tier = 10,000 units/day. Each video fetch costs ~3-5 units.
- **OpenAI API**: Check billing at https://platform.openai.com/usage
- **Anthropic API**: Check usage at https://console.anthropic.com
