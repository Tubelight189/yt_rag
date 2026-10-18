import argparse
import sys
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.logging import get_logger, setup_logging
from src.services.indexing import IndexingService

# Setup logging
setup_logging()
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Index a YouTube channel for RAG-based querying"
    )
    parser.add_argument(
        "channel",
        type=str,
        nargs="?",
        default="@TheDailyShow",
        help="YouTube channel handle (e.g., @veritasium) or channel ID",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=1,
        help="Maximum number of videos to index (default: 1)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing index before indexing",
    )

    args = parser.parse_args()
    channel_handle = args.channel
    max_videos = args.max_videos

    try:
        logger.info("Initializing indexing service...")
        indexing_service = IndexingService()

        # Clear index if requested
        if args.clear:
            logger.info("Clearing existing index...")
            indexing_service.clear_index()
            print("Index cleared.")

        # Index the channel
        logger.info(f"Starting to index channel: {channel_handle}")
        print(f"\nIndexing channel: {channel_handle}")
        print(f"Max videos: {max_videos}")
        print("This may take a few minutes...\n")

        stats = indexing_service.index_channel(
            channel_handle=channel_handle,
            max_videos=max_videos,
        )

        # Print results
        print("\n" + "=" * 60)
        print("Indexing Complete!")
        print("=" * 60)
        print(f"Channel Name:      {stats['channel_name']}")
        print(f"Channel ID:        {stats['channel_id']}")
        print(f"Videos Indexed:    {stats['videos_indexed']}")
        print(f"Videos Skipped:    {stats['videos_skipped']}")
        print(f"Total Chunks:      {stats['total_chunks']}")
        print("=" * 60)
        if stats["videos_indexed"]:
            print("\nYou can now ask questions about this channel's content!")
        else:
            print(
                "\nNo transcripts were indexed. Check the log for the reason, or use "
                "the app's Import Transcript tab with authorized transcript text."
            )
        print("Run the Gradio app with: make run")
        print("Or inspect the index with: make inspect")
        print()

    except KeyboardInterrupt:
        logger.info("Indexing interrupted by user")
        print("\nIndexing interrupted. Partial data may have been saved.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
