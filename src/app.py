"""Gradio UI for YouTube RAG system."""

import gradio as gr

from src.core.logging import get_logger, setup_logging
from src.services.chat import ChatService
from src.services.indexing import IndexingService
from src.youtube.videos import parse_video_id

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Initialize services
indexing_service = IndexingService()
chat_service = ChatService()


def index_channel(channel_handle: str, max_videos: int = 50) -> str:
    """Index a YouTube channel.

    Args:
        channel_handle: YouTube channel handle (e.g., @channelname)
        max_videos: Maximum number of videos to index

    Returns:
        Status message
    """
    try:
        if not channel_handle:
            return "Please enter a channel handle."

        logger.info(f"Starting to index channel: {channel_handle}")
        stats = indexing_service.index_channel(channel_handle, max_videos=max_videos)

        message = f"""
            --- Indexing Complete! ---

            Channel: {stats["channel_name"]}
            Videos Indexed: {stats["videos_indexed"]}
            Videos Skipped: {stats["videos_skipped"]}
            Total Chunks: {stats["total_chunks"]}

        """
        if stats["videos_indexed"]:
            message += "\nYou can now ask questions about this channel's content!"
        else:
            message += (
                "\nNo transcripts were indexed. Check the console for the reason, "
                "or use the Import Transcript tab with authorized transcript text."
            )
        return message.strip()

    except Exception as e:
        logger.error(f"Error indexing channel: {e}")
        return f"Error: {str(e)}"


def clear_index() -> str:
    """Clear the vector database.

    Returns:
        Status message
    """
    try:
        indexing_service.clear_index()
        return "Index cleared successfully!"
    except Exception as e:
        logger.error(f"Error clearing index: {e}")
        return f"Error: {str(e)}"


def index_video(video_url: str) -> str:
    try:
        if not video_url or not video_url.strip():
            return "Please enter a YouTube video URL."
        stats = indexing_service.index_video(video_url)
        return f"Video indexed successfully: {stats['video_title']} ({stats['video_id']})\nChunks: {stats['total_chunks']}"
    except Exception as error:
        logger.error("Error indexing video: %s", error)
        return f"Error: {error}\n\nIf captions are unavailable or YouTube blocks automated retrieval (IpBlocked/HTTP 429), use the Import Transcript tab."


def index_manual_transcript(video_url: str, video_title: str, transcript: str) -> str:
    """Index user-provided, authorized transcript text through the RAG pipeline."""
    try:
        video_id = video_url.strip().split("v=")[-1].split("&")[0] if video_url.strip() else "manual-transcript"
        stats = indexing_service.index_transcript(transcript, video_id, video_title)
        return (
            f"Transcript indexed successfully. Videos Indexed: {stats['videos_indexed']}; "
            f"Total Chunks: {stats['total_chunks']}"
        )
    except Exception as error:
        logger.error("Error indexing manual transcript: %s", error)
        return f"Error: {error}"


def chat(message: str, history: list[list[str]], video_scope: str = "") -> str:
    """Chat with the indexed content.

    Args:
        message: User message
        history: Chat history

    Returns:
        Bot response
    """
    try:
        if not message:
            return "Please enter a question."

        logger.info(f"Processing question: {message}")
        scope_id = parse_video_id(video_scope) if video_scope.strip() else None
        result = chat_service.ask(message, video_id=scope_id)

        # Format response with sources
        response = result["answer"]

        if result["sources"]:
            response += "\n\n**Sources:**\n"
            seen_videos = set()
            for source in result["sources"]:
                video_id = source["video_id"]
                if video_id not in seen_videos:
                    video_title = source["video_title"]
                    video_url = f"https://www.youtube.com/watch?v={video_id}"
                    response += f"- [{video_title}]({video_url})\n"
                    seen_videos.add(video_id)

        return response

    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        return f"Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="YouTube RAG System", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # YouTube RAG System
        Index YouTube channels and chat with their content using AI.
        """
    )

    with gr.Tab("Index Channel"):
        gr.Markdown(
            """
            ### Index a YouTube Channel
            Enter a YouTube channel handle to index its videos and transcripts.
            """
        )

        with gr.Row():
            channel_input = gr.Textbox(
                label="Channel Handle",
                placeholder="@channelname",
                info="Enter the YouTube channel handle (e.g., @veritasium)",
            )
            max_videos_input = gr.Slider(
                minimum=1,
                maximum=200,
                value=50,
                step=1,
                label="Max Videos",
                info="Maximum number of videos to index",
            )

        index_button = gr.Button("Index Channel", variant="primary")
        index_output = gr.Textbox(label="Status", lines=8)

        index_button.click(
            fn=index_channel,
            inputs=[channel_input, max_videos_input],
            outputs=index_output,
        )

        with gr.Row():
            clear_button = gr.Button("Clear Index", variant="stop")
            clear_output = gr.Textbox(label="Clear Status", lines=2)

        clear_button.click(fn=clear_index, outputs=clear_output)

    with gr.Tab("Index Video"):
        gr.Markdown("### Index one YouTube video\nPaste a public video URL. Captions are retrieved automatically when available.")
        video_url_input = gr.Textbox(label="YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
        video_index_button = gr.Button("Index Video", variant="primary")
        video_index_output = gr.Textbox(label="Status", lines=4)
        video_index_button.click(index_video, video_url_input, video_index_output)

    with gr.Tab("Chat"):
        gr.Markdown(
            """
            ### Ask Questions
            Chat with the indexed YouTube content. Make sure to index a channel first!
            """
        )

        chatbot = gr.Chatbot(
            label="Chat with YouTube Content",
            height=500,
        )
        msg = gr.Textbox(
            label="Your Question",
            placeholder="Ask a question about the indexed videos...",
        )
        video_scope = gr.Textbox(
            label="Scope to video (optional)",
            placeholder="Paste the indexed video's URL or 11-character ID",
            info="When set, retrieval uses only chunks from this video.",
        )

        with gr.Row():
            submit = gr.Button("Send", variant="primary")
            clear = gr.Button("Clear Chat")

        def respond(message, chat_history, selected_video_scope):
            """Handle a chat turn with the optional video scope."""
            bot_message = chat(message, chat_history, selected_video_scope or "")
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": bot_message})
            return "", chat_history

        submit.click(respond, [msg, chatbot, video_scope], [msg, chatbot])
        msg.submit(respond, [msg, chatbot, video_scope], [msg, chatbot])
        clear.click(lambda: [], None, chatbot, queue=False)

    with gr.Tab("Import Transcript"):
        gr.Markdown(
            """### Import an Authorized Transcript

            If YouTube blocks automated caption retrieval, paste transcript text you
            are authorized to use. It will be chunked, embedded, and stored exactly
            like a retrieved transcript.
            """
        )
        manual_url = gr.Textbox(label="Video URL (optional)")
        manual_title = gr.Textbox(label="Video title (optional)")
        manual_transcript = gr.Textbox(label="Transcript text", lines=14)
        manual_button = gr.Button("Index Transcript", variant="primary")
        manual_output = gr.Textbox(label="Status", lines=3)
        manual_button.click(
            fn=index_manual_transcript,
            inputs=[manual_url, manual_title, manual_transcript],
            outputs=manual_output,
        )

    gr.Markdown(
        """
        ---
        ### How to Use:
        1. Go to the "Index Channel" tab
        2. Enter a YouTube channel handle (e.g., @veritasium)
        3. Click "Index Channel" and wait for it to complete
        4. Go to the "Chat" tab
        5. Ask questions about the channel's content!
        """
    )


if __name__ == "__main__":
    logger.info("Starting YouTube RAG Gradio app")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
