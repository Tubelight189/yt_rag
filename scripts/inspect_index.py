"""Inspect the indexed videos in ChromaDB."""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import get_settings
from src.rag.vectorstore import ChromaVectorStore


def main():
    """Inspect the ChromaDB index."""
    print("=" * 60)
    print("ChromaDB Index Inspector")
    print("=" * 60)

    settings = get_settings()
    vector_store = ChromaVectorStore(persist_directory=settings.chroma_db_path)

    # Get collection stats
    collection = vector_store.collection
    count = collection.count()

    print(f"\nTotal documents indexed: {count}")

    if count == 0:
        print("\nNo documents found in the index.")
        print("Run 'make index CHANNEL=@channelname' to index a channel first.")
        return

    # Get all documents with metadata
    print("\nFetching documents...")
    results = collection.get(
        limit=count,
        include=['metadatas']
    )

    metadatas = results.get('metadatas', [])

    if not metadatas:
        print("No metadata found.")
        return

    # Aggregate by channel and video
    channels = {}
    videos = {}

    for metadata in metadatas:
        channel_id = metadata.get('channel_id', 'Unknown')
        channel_name = metadata.get('channel_name', 'Unknown')
        video_id = metadata.get('video_id', 'Unknown')
        video_title = metadata.get('video_title', 'Unknown')

        # Track channels
        if channel_id not in channels:
            channels[channel_id] = {
                'name': channel_name,
                'video_count': 0,
                'chunk_count': 0,
            }
        channels[channel_id]['chunk_count'] += 1

        # Track videos
        if video_id not in videos:
            videos[video_id] = {
                'title': video_title,
                'channel_name': channel_name,
                'channel_id': channel_id,
                'chunk_count': 0,
            }
            channels[channel_id]['video_count'] += 1

        videos[video_id]['chunk_count'] += 1

    # Display channel summary
    print("\n" + "=" * 60)
    print("INDEXED CHANNELS")
    print("=" * 60)

    for channel_id, info in channels.items():
        print(f"\nChannel: {info['name']}")
        print(f"   Channel ID: {channel_id}")
        print(f"   Videos: {info['video_count']}")
        print(f"   Total Chunks: {info['chunk_count']}")

    # Display video details
    print("\n" + "=" * 60)
    print("INDEXED VIDEOS")
    print("=" * 60)

    for video_id, info in sorted(videos.items(), key=lambda x: x[1]['channel_name']):
        print(f"\nVideo: {info['title']}")
        print(f"   Video ID: {video_id}")
        print(f"   URL: https://www.youtube.com/watch?v={video_id}")
        print(f"   Channel: {info['channel_name']}")
        print(f"   Chunks: {info['chunk_count']}")

    print("\n" + "=" * 60)
    print(f"TOTAL: {len(channels)} channel(s), {len(videos)} video(s), {count} chunk(s)")
    print("=" * 60)


if __name__ == "__main__":
    main()