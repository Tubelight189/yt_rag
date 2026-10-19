"""Safe extractors for non-YouTube sources."""
from hashlib import sha256
from ipaddress import ip_address
from urllib.parse import urljoin, urlparse, urlunparse
import socket

import fitz
import requests
from bs4 import BeautifulSoup


def stable_source_id(source_type: str, value: str) -> str:
    return f"{source_type}:{sha256(value.encode('utf-8')).hexdigest()[:24]}"


def extract_pdf(path: str) -> list[dict]:
    try:
        document = fitz.open(path)
    except Exception as exc:
        raise ValueError(f"Could not open PDF: {exc}") from exc
    pages = []
    try:
        for number, page in enumerate(document, 1):
            text = page.get_text("text").strip()
            if text:
                pages.append({"page_number": number, "text": text})
    finally:
        document.close()
    if not pages:
        raise ValueError("PDF contains no extractable text (scanned PDFs are not supported).")
    return pages


def normalize_url(value: str) -> str:
    parsed = urlparse((value or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Only public HTTP and HTTPS URLs are supported.")
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or "/", "", parsed.query, ""))


def _public_host(host: str) -> None:
    if not host or host.lower() in {"localhost", "localhost.localdomain"}:
        raise ValueError("Local and private network URLs are not allowed.")
    try:
        addresses = socket.getaddrinfo(host, None)
        for address in addresses:
            ip = ip_address(address[4][0])
            if not ip.is_global:
                raise ValueError("Local and private network URLs are not allowed.")
    except socket.gaierror as exc:
        raise ValueError("Could not resolve website host.") from exc


def fetch_website(url: str, timeout: int = 15, max_bytes: int = 5_000_000) -> dict:
    current = normalize_url(url)
    for _ in range(4):
        parsed = urlparse(current)
        _public_host(parsed.hostname or "")
        response = requests.get(current, timeout=timeout, stream=True, allow_redirects=False, headers={"User-Agent": "youtube-rag/1.0"})
        if response.is_redirect or response.is_permanent_redirect:
            current = normalize_url(urljoin(current, response.headers.get("location", "")))
            response.close()
            continue
        response.raise_for_status()
        if "text/html" not in response.headers.get("content-type", "").lower():
            raise ValueError("URL does not return HTML content.")
        content = response.content[:max_bytes + 1]
        response.close()
        if len(content) > max_bytes:
            raise ValueError("Website response is too large.")
        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        main = soup.find("main") or soup
        text = " ".join(main.get_text(" ", strip=True).split())
        if not text:
            raise ValueError("Website contains no readable text.")
        title = soup.title.get_text(strip=True) if soup.title else current
        canonical = soup.find("link", rel="canonical")
        final_url = normalize_url(canonical.get("href")) if canonical and canonical.get("href") else normalize_url(response.url)
        return {"url": final_url, "title": title, "text": text[:500_000]}
    raise ValueError("Too many redirects.")
