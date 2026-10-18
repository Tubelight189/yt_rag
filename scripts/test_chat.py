"""Test script for chat functionality."""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import get_settings
from src.rag.vectorstore import ChromaVectorStore
from src.services.chat import ChatService


def display_index_stats(vector_store: ChromaVectorStore):
    """Display stats about indexed content."""
    collection = vector_store.collection
    count = collection.count()

    print("\n" + "=" * 60)
    print("INDEXED CONTENT")
    print("=" * 60)

    if count == 0:
        print("\nNo documents found in the index.")
        print("Run 'make index CHANNEL=@channelname' to index a channel first.")
        return False

    # Get metadata to show channel/video info
    results = collection.get(limit=count, include=['metadatas'])
    metadatas = results.get('metadatas', [])

    # Aggregate by channel and video
    channels = {}
    videos = {}

    for metadata in metadatas:
        channel_id = metadata.get('channel_id', 'Unknown')
        channel_name = metadata.get('channel_name', 'Unknown')
        video_id = metadata.get('video_id', 'Unknown')
        video_title = metadata.get('video_title', 'Unknown')

        if channel_id not in channels:
            channels[channel_id] = {
                'name': channel_name,
                'video_count': 0,
                'chunk_count': 0,
            }
        channels[channel_id]['chunk_count'] += 1

        if video_id not in videos:
            videos[video_id] = {
                'title': video_title,
                'channel_name': channel_name,
            }
            channels[channel_id]['video_count'] += 1

    print(f"\nTotal documents: {count}")
    print(f"Channels indexed: {len(channels)}")
    print(f"Videos indexed: {len(videos)}")

    print("\nChannels:")
    for channel_id, info in channels.items():
        print(f"  - {info['name']}: {info['video_count']} videos, {info['chunk_count']} chunks")

    return True


def display_result(result: dict):
    """Display chat result nicely formatted."""
    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(f"\n{result['answer']}\n")

    sources = result.get('sources', [])
    if sources:
        print("=" * 60)
        print("SOURCES")
        print("=" * 60)

        # Group sources by video
        video_sources = {}
        for source in sources:
            video_id = source['video_id']
            if video_id not in video_sources:
                video_sources[video_id] = source

        for i, source in enumerate(video_sources.values(), 1):
            print(f"\n{i}. {source['video_title']}")
            print(f"   Channel: {source['channel_name']}")
            print(f"   URL: https://www.youtube.com/watch?v={source['video_id']}")
            print(f"   Relevance score: {source['score']:.4f}")

    print("\n" + "=" * 60)


def interactive_chat(chat_service: ChatService):
    """Run interactive chat loop."""
    print("\n" + "=" * 60)
    print("INTERACTIVE CHAT")
    print("=" * 60)
    print("\nType your questions below.")
    print("Commands:")
    print("  - Type 'quit' or 'exit' to stop")
    print("  - Type 'clear' to clear the screen")
    print("=" * 60)

    while True:
        try:
            question = input("\nYou: ").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit']:
                print("\nExiting chat. Goodbye!")
                break

            if question.lower() == 'clear':
                print("\033[H\033[J", end="")
                continue

            # Ask question
            result = chat_service.ask(question)

            # Display result
            display_result(result)

        except KeyboardInterrupt:
            print("\n\nExiting chat. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main():
    """Test chat functionality."""
    print("=" * 60)
    print("Testing Chat Functionality")
    print("=" * 60)

    try:
        settings = get_settings()

        # Check if we have necessary API keys
        if settings.llm_provider == "openai" and not settings.openai_api_key:
            print("\nError: OpenAI API key not set.")
            print("Set OPENAI_API_KEY in your .env file")
            return

        if settings.llm_provider == "anthropic" and not settings.anthropic_api_key:
            print("\nError: Anthropic API key not set.")
            print("Set ANTHROPIC_API_KEY in your .env file")
            return

        # Initialize chat service
        print("\nInitializing chat service...")
        chat_service = ChatService(settings=settings)

        # Display index stats
        has_data = display_index_stats(chat_service.vector_store)

        if not has_data:
            return

        # Show settings
        print("\n" + "=" * 60)
        print("SETTINGS")
        print("=" * 60)
        print(f"LLM Provider: {settings.llm_provider}")
        print(f"LLM Model: {settings.llm_model}")
        print(f"Embedding: {'Local' if settings.should_local_embed else 'OpenAI'}")

        # Run interactive chat
        interactive_chat(chat_service)

    except Exception as e:
        print(f"\nFailed to initialize chat service: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
