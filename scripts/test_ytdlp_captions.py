"""Test script for extracting captions using yt-dlp."""

import json
import sys
from pathlib import Path

import yt_dlp

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_transcript_ytdlp(video_id: str, languages: list[str] | None = None) -> str:
    """Get transcript using yt-dlp.

    Args:
        video_id: YouTube video ID
        languages: Preferred languages (default: ['en'])

    Returns:
        Video transcript as text

    Raises:
        Exception: If transcript extraction fails
    """
    if languages is None:
        languages = ['en']

    video_url = f"https://www.youtube.com/watch?v={video_id}"

    # Configure yt-dlp options
    ydl_opts = {
        'skip_download': True,  # Don't download video
        'writesubtitles': True,  # Write subtitle file
        'writeautomaticsub': True,  # Write automatic captions
        'subtitleslangs': ['en'],  # Preferred languages
        'subtitlesformat': 'json3',  # JSON format (easier to parse)
        'quiet': False,  # Suppress output
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info
            info = ydl.extract_info(video_url, download=False)

            # Check for subtitles
            subtitles = info.get('subtitles', {})
            automatic_captions = info.get('automatic_captions', {})

            print(f"\nVideo: {info.get('title')}")
            print(f"Duration: {info.get('duration')}s")
            print(f"\nAvailable subtitles: {list(subtitles.keys())}")
            print(f"Available auto captions: {list(automatic_captions.keys())}")

            # Try to get captions in preferred language
            captions = None

            # First try manual subtitles
            for lang in languages:
                if lang in subtitles:
                    captions = subtitles[lang]
                    print(f"\n Using manual subtitles: {lang}")
                    break

            # Fall back to automatic captions
            if not captions:
                for lang in languages:
                    if lang in automatic_captions:
                        captions = automatic_captions[lang]
                        print(f"\n Using automatic captions: {lang}")
                        break

            # Try English as fallback
            if not captions and 'en' in automatic_captions:
                captions = automatic_captions['en']
                print("\n Using automatic captions (fallback): en")

            if not captions:
                raise Exception(f"No captions found for video {video_id}")

            # Find JSON3 format
            json3_caption = None
            for caption_format in captions:
                if caption_format.get('ext') == 'json3':
                    json3_caption = caption_format
                    break

            if not json3_caption:
                raise Exception("JSON3 caption format not available")

            # Download the caption data
            caption_url = json3_caption['url']
            caption_data = ydl.urlopen(caption_url).read().decode('utf-8')
            caption_json = json.loads(caption_data)

            # Extract text from JSON
            transcript_parts = []
            events = caption_json.get('events', [])

            for event in events:
                segs = event.get('segs')
                if segs:
                    for seg in segs:
                        text = seg.get('utf8', '').strip()
                        if text:
                            transcript_parts.append(text)

            full_transcript = ' '.join(transcript_parts)

            print(f"\nTranscript length: {len(full_transcript)} characters")
            print(f"First 200 chars: {full_transcript[:200]}...")

            return full_transcript

    except Exception as e:
        print(f"\nError: {e}")
        raise


def main():
    """Test caption extraction."""
    print("=" * 60)
    print("Testing yt-dlp Caption Extraction")
    print("=" * 60)

    # Test with different videos
    test_videos = [
        "oT896h76s-Y",  # Rick Astley - Never Gonna Give You Up
    ]

    for video_id in test_videos:
        print(f"\n{'=' * 60}")
        print(f"Testing video: {video_id}")
        print("=" * 60)

        try:
            transcript = get_transcript_ytdlp(video_id)
            print(f"\nSUCCESS! Got {len(transcript)} characters")
        except Exception as e:
            print(f"\nFAILED: {e}")

        print("\n" + "=" * 60)
        input("\nPress Enter to test next video...")


if __name__ == "__main__":
    main()
