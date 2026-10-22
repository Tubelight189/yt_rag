"""Gradio UI for YouTube RAG system."""
import os
from pathlib import Path
from functools import partial

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


def index_channel(channel_handle: str, max_videos: int = 5) -> str:
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
    return gr.update(choices=scope_choices(), value="", interactive=True)


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
            seen_sources = set()
            for source in result["sources"]:
                video_id = source["video_id"]
                source_key = source.get("source_id") or video_id
                if source_key and source_key not in seen_sources:
                    video_title = source.get("video_title", "Unknown")
                    source_type = source.get("source_type", "source").replace("_", " ").title()
                    if video_id:
                        label = video_title if video_title != "Unknown" else source_type
                        video_url = f"https://www.youtube.com/watch?v={video_id}"
                        response += f"- [{label}]({video_url})\n"
                    else:
                        # Non-YouTube source metadata does not include a public URL.
                        source_ref = source.get("source_id") or source_type
                        response += f"- {source_type} source (`{source_ref}`)\n"
                    seen_sources.add(source_key)

        return response

    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        return f"Error: {str(e)}"


# Frontend-only helpers keep the service callbacks above unchanged.
def refresh_scope_pair():
    return refresh_scope(), refresh_scope()


def index_channel_ui(handle: str, limit: int):
    return index_channel(handle, limit), *refresh_scope_pair()


def index_video_ui(url: str):
    return index_video(url), *refresh_scope_pair()


def index_pdf_ui(path: str):
    return index_pdf(path), *refresh_scope_pair()


def index_website_ui(url: str):
    return index_website(url), *refresh_scope_pair()


def index_transcript_ui(url: str, title: str, transcript: str):
    return index_manual_transcript(url, title, transcript), *refresh_scope_pair()


def refresh_library_ui():
    return refresh_scope_pair()


def delete_source_ui(source_id: str, confirmed: bool):
    if not confirmed:
        status = "Confirm source deletion using the checkbox first."
    else:
        status, _ = delete_selected_source(source_id)
    return status, *refresh_scope_pair(), gr.update(value=False)


def clear_library_ui(confirmed: bool):
    if not confirmed:
        status = "Confirm clearing the full index using the checkbox first."
    else:
        status, _ = clear_index_and_refresh()
    return status, *refresh_scope_pair(), gr.update(value=False)


def respond(message, chat_history, selected_source):
    """Append one user and assistant turn in Gradio's messages format."""
    history = list(chat_history or [])
    if not message or not message.strip():
        return "", history
    answer = chat(message, history, selected_source or "")
    history.extend([
        {"role": "user", "content": message.strip()},
        {"role": "assistant", "content": answer},
    ])
    return "", history


NAV_ITEMS = [
    ("Dashboard", "⌂"), ("AI Chat", "✦"),
    ("YouTube Channel", "▶"), ("YouTube Video", "▷"),
    ("PDF Documents", "▤"), ("Website Analyzer", "◎"),
    ("Transcript Import", "≋"), ("Manage Sources", "⚙"),
]
PAGE_NAMES = [name for name, _ in NAV_ITEMS]


def select_page(page: str):
    updates = [gr.update(visible=(name == page)) for name in PAGE_NAMES]
    updates.extend(gr.update(variant="primary" if name == page else "secondary") for name in PAGE_NAMES)
    return updates


with (Path(__file__).resolve().parent / "assets" / "styles.css").open(encoding="utf-8") as css_file:
    APP_CSS = css_file.read()


with gr.Blocks(title="NEXUS AI | Intelligent Knowledge Companion") as demo:
    with gr.Sidebar(position="left", width=270, elem_id="nexus-sidebar"):
        with gr.Column(elem_classes="sidebar-navigation"):
            gr.HTML("""
                <div class="brand-lockup"><div class="brand-symbol"><span>✦</span></div>
                <div><div class="brand-name">NEXUS <span>AI</span></div><div class="brand-subtitle">KNOWLEDGE, CONNECTED</div></div></div>
            """)
            gr.Markdown("MAIN", elem_classes="nav-section-label")
            nav_buttons = {}
            for nav_name, nav_icon in NAV_ITEMS[:2]:
                nav_buttons[nav_name] = gr.Button(f"{nav_icon}   {nav_name}", variant="primary" if nav_name == "Dashboard" else "secondary", elem_classes="nav-button")
            gr.Markdown("KNOWLEDGE SOURCES", elem_classes="nav-section-label")
            for nav_name, nav_icon in NAV_ITEMS[2:7]:
                nav_buttons[nav_name] = gr.Button(f"{nav_icon}   {nav_name}", variant="secondary", elem_classes="nav-button")
            gr.Markdown("MANAGEMENT", elem_classes="nav-section-label")
            nav_buttons["Manage Sources"] = gr.Button("⚙   Manage Sources", variant="secondary", elem_classes="nav-button")
        gr.HTML("<div class='sidebar-foot'><span class='status-dot'></span><span>Your Private Knowledge Space</span></div>")

    with gr.Column(elem_id="nexus-main"):
        with gr.Group(visible=True, elem_id="nexus-dashboard", elem_classes=["page-view", "nexus-page-background"]) as dashboard_page:
            gr.HTML("""
                <section class="hero-card"><div class="hero-glow hero-glow-one"></div><div class="hero-glow hero-glow-two"></div>
                  <div class="hero-content"><div class="eyebrow"><span class="live-dot"></span> YOUR INTELLIGENT KNOWLEDGE COMPANION</div>
                    <h1>Transform Content<br><span>Into Intelligence</span></h1>
                    <p>Unlock insights from YouTube videos, documents, and websites using the power of AI. Upload, analyze, and interact with your knowledge effortlessly.</p>
                  </div><div class="hero-orbit" aria-hidden="true"><div class="orbit-ring ring-one"></div><div class="orbit-ring ring-two"></div><div class="orbit-core">✦</div><i class="orbit-node node-one"></i><i class="orbit-node node-two"></i><i class="orbit-node node-three"></i></div>
                </section>
            """)
            gr.Markdown("###      Welcome to NEXUS AI\nYour AI-powered workspace for intelligent content exploration.", elem_classes="welcome-copy")
            with gr.Row(elem_classes="overview-row"):
                with gr.Column(scale=3, elem_classes="source-count-card"):
                    gr.Markdown("**KNOWLEDGE LIBRARY**")
                    source_count_display = gr.Markdown("**Indexed Sources**", elem_classes="source-count-value")
                with gr.Column(scale=2, elem_classes="overview-hint"):
                    gr.Markdown("**A GROUNDED AI WORKSPACE**\n\nAnswers are drawn from the sources you choose.")
            gr.Markdown("### Explore your knowledge tools", elem_classes="cards-heading")
            with gr.Row(elem_classes="feature-grid"):
                with gr.Column(elem_classes="feature-card cyan-card"):
                    gr.HTML("<div class='feature-icon'>▶</div><div class='feature-kicker'>VIDEO LIBRARY</div><h3>YouTube Intelligence</h3><p>Analyze YouTube videos and channels.</p>")
                    dashboard_channel = gr.Button("Explore channel indexing  ↗", variant="secondary")
                with gr.Column(elem_classes="feature-card purple-card"):
                    gr.HTML("<div class='feature-icon'>▤</div><div class='feature-kicker'>DOCUMENTS</div><h3>Document Intelligence</h3><p>Upload PDFs and extract valuable insights.</p>")
                    dashboard_pdf = gr.Button("Explore PDF tools  ↗", variant="secondary")
                with gr.Column(elem_classes="feature-card blue-card"):
                    gr.HTML("<div class='feature-icon'>◎</div><div class='feature-kicker'>WEB CONTENT</div><h3>Website Intelligence</h3><p>Transform web content into searchable knowledge.</p>")
                    dashboard_website = gr.Button("Explore website tools  ↗", variant="secondary")
                with gr.Column(elem_classes="feature-card violet-card"):
                    gr.HTML("<div class='feature-icon'>✦</div><div class='feature-kicker'>ASK YOUR LIBRARY</div><h3>AI Knowledge Chat</h3><p>Ask questions and receive intelligent answers.</p>")
                    dashboard_chat = gr.Button("Start a conversation  ↗", variant="secondary")

        with gr.Group(visible=False, elem_classes=["page-view", "nexus-page-background"]) as chat_page:
            gr.HTML("<div class='page-heading'><div class='eyebrow'>NEXUS CONVERSATION</div><h1>AI Knowledge Chat</h1><p>Explore your indexed sources with grounded answers and references.</p></div>")
            gr.Markdown("RETRIEVAL SCOPE", elem_classes="source-selector-label")
            gr.Markdown("Search across all sources or focus on one.", elem_classes="source-selector-description")
            chat_scope = gr.Dropdown(choices=scope_choices(), value="", container=False, elem_id="nexus-source-selector", elem_classes=["nexus-input", "nexus-source-selector"])
            chatbot = gr.Chatbot(label="Conversation", height=490, layout="bubble", autoscroll=True,
                placeholder="<div class='chat-empty'><div class='chat-empty-icon'>✦</div><h2>Hello! I'm NEXUS AI.</h2><p>Ask me anything about your indexed knowledge sources.</p></div>",
                elem_id="nexus-chatbot", elem_classes="nexus-chatbot")
            with gr.Row(elem_classes="chat-compose"):
                msg = gr.Textbox(label="", placeholder="Ask a question about your knowledge…", lines=2, scale=8, elem_classes="nexus-input")
                with gr.Column(scale=1, min_width=145):
                    submit = gr.Button("Send  ↑", variant="primary", elem_classes="send-button")
                    clear_chat = gr.Button("Clear chat", variant="secondary")
            submit.click(respond, [msg, chatbot, chat_scope], [msg, chatbot], show_progress="minimal")
            msg.submit(respond, [msg, chatbot, chat_scope], [msg, chatbot], show_progress="minimal")
            clear_chat.click(lambda: ("", []), outputs=[msg, chatbot], queue=False)

        with gr.Group(visible=False, elem_classes=["page-view", "nexus-page-background"]) as channel_page:
            gr.HTML("<div class='page-heading'><div class='eyebrow'>KNOWLEDGE SOURCES / YOUTUBE</div><h1>YouTube Channel</h1><p>Bring a channel’s video library into your searchable knowledge space.</p></div>")
            with gr.Column(elem_classes="glass-panel source-panel"):
                gr.HTML("<div class='source-panel-icon cyan-icon'>▶</div><h2>Index a YouTube channel</h2><p>Enter a channel handle and choose how many recent videos to scan.</p>")
                channel_input = gr.Textbox(label="CHANNEL HANDLE", placeholder="@veritasium", info="Use the channel handle, for example @veritasium.", elem_classes="nexus-input")
                max_videos_input = gr.Slider(minimum=1, maximum=200, value=5, step=1, label="MAXIMUM VIDEOS")
                index_button = gr.Button("Index Channel  ↗", variant="primary", elem_classes="gradient-action")
                index_output = gr.Textbox(label="INDEXING STATUS", lines=4, interactive=False)

        with gr.Group(visible=False, elem_classes=["page-view", "nexus-page-background"]) as video_page:
            gr.HTML("<div class='page-heading'><div class='eyebrow'>KNOWLEDGE SOURCES / YOUTUBE</div><h1>YouTube Video</h1><p>Turn one video into a source you can search and ask questions about.</p></div>")
            with gr.Column(elem_classes="glass-panel source-panel video-panel"):
                gr.HTML("<div class='source-panel-icon red-icon'>▷</div><h2>Analyze a video</h2><p>Paste a public YouTube video URL. Captions are indexed when available.</p>")
                video_url_input = gr.Textbox(label="YOUTUBE VIDEO URL", placeholder="https://www.youtube.com/watch?v=…", elem_classes="nexus-input")
                video_index_button = gr.Button("Analyze Video  ↗", variant="primary", elem_classes="gradient-action")
                video_index_output = gr.Textbox(label="ANALYSIS STATUS", lines=4, interactive=False)

        with gr.Group(visible=False, elem_classes=["page-view", "nexus-page-background"]) as pdf_page:
            gr.HTML("<div class='page-heading'><div class='eyebrow'>KNOWLEDGE SOURCES / DOCUMENTS</div><h1>PDF Documents</h1><p>Extract searchable knowledge from your course notes and reference documents.</p></div>")
            with gr.Column(elem_classes="glass-panel source-panel upload-panel"):
                gr.HTML("<div class='source-panel-icon purple-icon'>▤</div><h2>Process a document</h2><p>Choose a PDF to add its extractable text to your knowledge library.</p>")
                pdf_input = gr.File(label="DROP A PDF HERE OR BROWSE", file_types=[".pdf"], type="filepath", elem_classes="pdf-dropzone")
                pdf_button = gr.Button("Process Document  ↗", variant="primary", elem_classes="gradient-action")
                pdf_output = gr.Textbox(label="PROCESSING STATUS", lines=4, interactive=False)

        with gr.Group(visible=False, elem_classes=["page-view", "nexus-page-background"]) as website_page:
            gr.HTML("<div class='page-heading'><div class='eyebrow'>KNOWLEDGE SOURCES / WEB</div><h1>Website Analyzer</h1><p>Make an individual public webpage part of your searchable knowledge.</p></div>")
            with gr.Column(elem_classes="glass-panel source-panel"):
                gr.HTML("<div class='source-panel-icon blue-icon'>◎</div><h2>Analyze a webpage</h2><p>Add an article or public webpage to the NEXUS knowledge space.</p>")
                website_input = gr.Textbox(label="PUBLIC WEBSITE URL", placeholder="https://example.com/article", elem_classes="nexus-input")
                website_button = gr.Button("Analyze Website  ↗", variant="primary", elem_classes="gradient-action")
                website_output = gr.Textbox(label="ANALYSIS STATUS", lines=4, interactive=False)

        with gr.Group(visible=False, elem_classes=["page-view", "nexus-page-background"]) as transcript_page:
            gr.HTML("<div class='page-heading'><div class='eyebrow'>KNOWLEDGE SOURCES / IMPORT</div><h1>Transcript Import</h1><p>Use transcript text you are authorized to add to your searchable library.</p></div>")
            with gr.Column(elem_classes="glass-panel source-panel transcript-panel"):
                gr.HTML("<div class='source-panel-icon violet-icon'>≋</div><h2>Import a transcript</h2><p>Add optional video details, then paste the transcript below.</p>")
                manual_url = gr.Textbox(label="YOUTUBE VIDEO URL · OPTIONAL", placeholder="https://www.youtube.com/watch?v=…", elem_classes="nexus-input")
                manual_title = gr.Textbox(label="VIDEO TITLE · OPTIONAL", placeholder="Lecture 01 — Introduction", elem_classes="nexus-input")
                manual_transcript = gr.Textbox(label="TRANSCRIPT TEXT", lines=16, placeholder="Paste transcript text here…", elem_classes="nexus-input")
                manual_button = gr.Button("Import Transcript  ↗", variant="primary", elem_classes="gradient-action")
                manual_output = gr.Textbox(label="IMPORT STATUS", lines=4, interactive=False)

        with gr.Group(visible=False, elem_classes=["page-view", "nexus-page-background"]) as management_page:
            gr.HTML("<div class='page-heading'><div class='eyebrow'>MANAGEMENT</div><h1>Manage Sources</h1><p>Refresh, remove a selected source, or clear the indexed library.</p></div>")
            with gr.Column(elem_classes="glass-panel source-panel"):
                gr.Markdown("INDEXED SOURCES", elem_classes="source-selector-label")
                gr.Markdown("Choose a source to remove.", elem_classes="source-selector-description")
                manage_scope = gr.Dropdown(choices=scope_choices(), value="", container=False, elem_id="nexus-manage-source-selector", elem_classes=["nexus-input", "nexus-source-selector"])
                refresh_button = gr.Button("Refresh Sources  ↻", variant="secondary")
                confirm_actions = gr.Checkbox(label="I understand these actions permanently remove indexed content.", value=False)
                with gr.Row():
                    delete_button = gr.Button("Delete Selected Source", variant="secondary")
                    clear_button = gr.Button("Clear Entire Index", variant="stop", elem_id="clear-entire-index-btn")
                management_output = gr.Textbox(label="MANAGEMENT STATUS", lines=3, interactive=False)

            index_button.click(index_channel_ui, [channel_input, max_videos_input], [index_output, chat_scope, manage_scope], show_progress="minimal")
            video_index_button.click(index_video_ui, video_url_input, [video_index_output, chat_scope, manage_scope], show_progress="minimal")
            pdf_button.click(index_pdf_ui, pdf_input, [pdf_output, chat_scope, manage_scope], show_progress="minimal")
            website_button.click(index_website_ui, website_input, [website_output, chat_scope, manage_scope], show_progress="minimal")
            manual_button.click(index_transcript_ui, [manual_url, manual_title, manual_transcript], [manual_output, chat_scope, manage_scope], show_progress="minimal")
            refresh_button.click(refresh_library_ui, outputs=[chat_scope, manage_scope], queue=False)
            delete_button.click(delete_source_ui, [manage_scope, confirm_actions], [management_output, chat_scope, manage_scope, confirm_actions], show_progress="minimal")
            clear_button.click(clear_library_ui, confirm_actions, [management_output, chat_scope, manage_scope, confirm_actions], show_progress="minimal")

    pages = [dashboard_page, chat_page, channel_page, video_page, pdf_page, website_page, transcript_page, management_page]
    nav_components = [nav_buttons[name] for name in PAGE_NAMES]
    for nav_name in PAGE_NAMES:
        nav_buttons[nav_name].click(partial(select_page, nav_name), outputs=pages + nav_components, queue=False)
    dashboard_channel.click(lambda: select_page("YouTube Channel"), outputs=pages + nav_components, queue=False)
    dashboard_pdf.click(lambda: select_page("PDF Documents"), outputs=pages + nav_components, queue=False)
    dashboard_website.click(lambda: select_page("Website Analyzer"), outputs=pages + nav_components, queue=False)
    dashboard_chat.click(lambda: select_page("AI Chat"), outputs=pages + nav_components, queue=False)


if __name__ == "__main__":
    logger.info("Starting YouTube RAG Gradio app")
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
        theme=gr.themes.Base(),
        css=APP_CSS,
    )
