"""Safe extractors for non-YouTube sources."""
from hashlib import sha256
from ipaddress import ip_address
from urllib.parse import urljoin, urlparse, urlunparse
import socket

import fitz
import requests
from bs4 import BeautifulSoup


_REDIRECT_CODES = {301, 302, 303, 307, 308}
_BOILERPLATE_TAGS = (
    "script", "style", "nav", "footer", "header", "aside", "form",
    "noscript", "svg", "canvas", "template",
)


def stable_source_id(source_type: str, value: str) -> str:
    """Return a deterministic, namespaced identifier for a source."""
    return f"{source_type}:{sha256(value.encode('utf-8')).hexdigest()[:24]}"


def extract_pdf(path: str) -> list[dict]:
    """Extract non-empty text page-by-page; scanned PDFs require OCR (not supported)."""
    try:
        document = fitz.open(path)
    except Exception as exc:
        raise ValueError(f"Could not open PDF: {exc}") from exc

    pages = []
    try:
        if not getattr(document, "is_pdf", True):
            raise ValueError("The uploaded file is not a valid PDF.")
        for number, page in enumerate(document, 1):
            try:
                text = page.get_text("text").strip()
            except Exception as exc:
                raise ValueError(f"Could not extract text from PDF page {number}: {exc}") from exc
            if text:
                pages.append({"page_number": number, "text": text})
    finally:
        document.close()

    if not pages:
        raise ValueError("PDF contains no extractable text (scanned PDFs are not supported).")
    return pages


def normalize_url(value: str) -> str:
    """Normalize an absolute HTTP(S) URL and reject ambiguous/unsafe URL forms."""
    try:
        parsed = urlparse((value or "").strip())
        hostname = parsed.hostname
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise ValueError("Please provide a valid public HTTP or HTTPS URL.") from exc

    if parsed.scheme.lower() not in {"http", "https"} or not hostname or not parsed.netloc:
        raise ValueError("Only public HTTP and HTTPS URLs are supported.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URLs containing embedded credentials are not allowed.")
    expected_port = 443 if parsed.scheme.lower() == "https" else 80
    if port is not None and port != expected_port:
        raise ValueError("Only standard HTTP (80) and HTTPS (443) ports are supported.")

    try:
        normalized_host = hostname.encode("idna").decode("ascii").lower().rstrip(".")
    except UnicodeError as exc:
        raise ValueError("The website hostname is invalid.") from exc
    if not normalized_host:
        raise ValueError("Please provide a valid website hostname.")

    # Preserve IPv6 syntax when rebuilding the netloc.
    netloc_host = f"[{normalized_host}]" if ":" in normalized_host else normalized_host
    netloc = netloc_host if port is None else f"{netloc_host}:{port}"
    return urlunparse((parsed.scheme.lower(), netloc, parsed.path or "/", "", parsed.query, ""))


def _public_host(host: str) -> None:
    """Reject non-global IPs and hostnames resolving to any non-global address."""
    if not host or host.lower().rstrip(".") in {"localhost", "localhost.localdomain"}:
        raise ValueError("Local and private network URLs are not allowed.")

    try:
        try:
            addresses = {ip_address(host)}
        except ValueError:
            addresses = {
                ip_address(item[4][0])
                for item in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
            }
        if not addresses or any(not address.is_global for address in addresses):
            raise ValueError("Local and private network URLs are not allowed.")
    except socket.gaierror as exc:
        raise ValueError("Could not resolve website host.") from exc


def fetch_website(url: str, timeout: int = 15, max_bytes: int = 5_000_000) -> dict:
    """Fetch one public HTML page with bounded redirects and streamed body size.

    DNS is checked for every redirect hop. Requests still performs its own DNS
    lookup when opening the socket; without IP-pinned transport this is a
    best-effort SSRF defense and does not eliminate DNS-rebinding races.
    """
    if max_bytes <= 0:
        raise ValueError("Maximum website response size must be positive.")

    current = normalize_url(url)
    session = requests.Session()
    session.trust_env = False  # Do not route security-sensitive fetches through ambient proxies.
    try:
        for redirect_count in range(6):
            parsed = urlparse(current)
            _public_host(parsed.hostname or "")

            response = session.get(
                current,
                timeout=timeout,
                stream=True,
                allow_redirects=False,
                headers={"User-Agent": "QueryWise/1.0"},
            )
            try:
                if response.status_code in _REDIRECT_CODES:
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("Website returned a redirect without a destination.")
                    if redirect_count == 5:
                        raise ValueError("Too many redirects.")
                    current = normalize_url(urljoin(current, location))
                    continue

                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if "text/html" not in content_type:
                    raise ValueError("URL does not return HTML content.")

                declared_size = response.headers.get("content-length")
                if declared_size:
                    try:
                        if int(declared_size) > max_bytes:
                            raise ValueError("Website response is too large.")
                    except ValueError as exc:
                        if str(exc) == "Website response is too large.":
                            raise
                        raise ValueError("Website returned an invalid content length.") from exc

                body = bytearray()
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    if len(body) + len(chunk) > max_bytes:
                        raise ValueError("Website response is too large.")
                    body.extend(chunk)

                # Detect DNS changes across the request as a defense-in-depth check.
                _public_host(parsed.hostname or "")
                soup = BeautifulSoup(bytes(body), "html.parser")
                for tag in soup(_BOILERPLATE_TAGS):
                    tag.decompose()

                main = soup.find("main") or soup.find("article") or soup.body or soup
                text = " ".join(main.get_text(" ", strip=True).split())
                if not text:
                    raise ValueError("Website contains no readable text.")

                title = soup.title.get_text(" ", strip=True) if soup.title else current
                return {
                    "url": current,
                    "title": title or current,
                    "text": text[:500_000],
                }
            finally:
                response.close()
        raise ValueError("Too many redirects.")
    finally:
        session.close()
