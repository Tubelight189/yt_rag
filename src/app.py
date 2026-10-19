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


def clear_index_and_refresh():
    return clear_index(), refresh_scope()


def delete_selected_source(source_id: str):
    if not source_id:
        return "Select a source first.", refresh_scope()
    try:
        indexing_service.delete_source(source_id)
        return "Source deleted successfully.", refresh_scope()
    except Exception as error:
        return f"Error: {error}", refresh_scope()


def index_video(video_url: str) -> str:
    try:
        if not video_url or not video_url.strip():
            return "Please enter a YouTube video URL."
        stats = indexing_service.index_video(video_url)
        return f"Video indexed successfully: {stats['video_title']} ({stats['video_id']})\nChunks: {stats['total_chunks']}"
    except Exception as error:
        logger.error("Error indexing video: %s", error)
        return f"Error: {error}\n\nIf captions are unavailable or YouTube blocks automated retrieval (IpBlocked/HTTP 429), use the Import Transcript tab."


def index_pdf(file_path: str) -> str:
    try:
        if not file_path:
            return "Please upload a PDF file."
        stats = indexing_service.index_pdf(file_path)
        return f"PDF indexed successfully: {stats['source_name']}\nChunks: {stats['total_chunks']}"
    except Exception as error:
        return f"Error: {error}"


def index_website(url: str) -> str:
    try:
        if not url or not url.strip():
            return "Please enter a website URL."
        stats = indexing_service.index_website(url)
        return f"Website indexed successfully: {stats['source_name']}\nURL: {stats['source_url']}\nChunks: {stats['total_chunks']}"
    except Exception as error:
        return f"Error: {error}"


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


def scope_choices() -> list[tuple[str, str]]:
    choices = [("All Sources", "")]
    seen = set()
    for source in indexing_service.list_sources():
        source_id = source.get("source_id")
        if not source_id or source_id in seen:
            continue
        seen.add(source_id)
        source_type = source.get("source_type", "source").replace("_", " ").title()
        name = source.get("source_name") or source.get("video_title") or source_id
        choices.append((f"{name} — {source_type}", source_id))
    return choices


def refresh_scope():
    return gr.Dropdown(choices=scope_choices(), value="", interactive=True)


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
        selected_source = video_scope.strip() or None
        valid_ids = {value for _, value in scope_choices() if value}
        if selected_source and selected_source not in valid_ids:
            return "The selected source is no longer indexed. Refresh the source list and select a valid source."
        result = chat_service.ask(message, source_id=selected_source)

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

    scope_dropdown = gr.Dropdown(
        label="Retrieval scope",
        choices=scope_choices(),
        value="",
        info="All Sources or one indexed source.",
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

        clear_button.click(fn=clear_index_and_refresh, outputs=[clear_output, scope_dropdown])

    with gr.Tab("Index Video"):
        gr.Markdown("### Index one YouTube video\nPaste a public video URL. Captions are retrieved automatically when available.")
        video_url_input = gr.Textbox(label="YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
        video_index_button = gr.Button("Index Video", variant="primary")
        video_index_output = gr.Textbox(label="Status", lines=4)
        video_index_button.click(lambda url: (index_video(url), refresh_scope()), video_url_input, [video_index_output, scope_dropdown])

    with gr.Tab("Index PDF"):
        pdf_input = gr.File(label="PDF file", file_types=[".pdf"], type="filepath")
        pdf_button = gr.Button("Index PDF", variant="primary")
        pdf_output = gr.Textbox(label="Status", lines=3)
        pdf_button.click(lambda path: (index_pdf(path), refresh_scope()), pdf_input, [pdf_output, scope_dropdown])

    with gr.Tab("Index Website"):
        website_input = gr.Textbox(label="Public webpage URL", placeholder="https://example.com/article")
        website_button = gr.Button("Index Website", variant="primary")
        website_output = gr.Textbox(label="Status", lines=3)
        website_button.click(lambda url: (index_website(url), refresh_scope()), website_input, [website_output, scope_dropdown])

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
        with gr.Row():
            submit = gr.Button("Send", variant="primary")
            clear = gr.Button("Clear Chat")

        def respond(message, chat_history, selected_video_scope):
            """Handle a chat turn with the optional video scope."""
            bot_message = chat(message, chat_history, selected_video_scope or "")
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": bot_message})
            return "", chat_history

        submit.click(respond, [msg, chatbot, scope_dropdown], [msg, chatbot])
        msg.submit(respond, [msg, chatbot, scope_dropdown], [msg, chatbot])
        clear.click(lambda: [], None, chatbot, queue=False)

        delete_button = gr.Button("Delete Selected Source", variant="stop")
        delete_output = gr.Textbox(label="Source Status", lines=2)
        delete_button.click(delete_selected_source, scope_dropdown, [delete_output, scope_dropdown])

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
            fn=lambda url, title, transcript: (index_manual_transcript(url, title, transcript), refresh_scope()),
            inputs=[manual_url, manual_title, manual_transcript],
            outputs=[manual_output, scope_dropdown],
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
