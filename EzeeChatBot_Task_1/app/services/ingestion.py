"""
Ingestion service — pulls raw text from a URL or a raw text payload.
Handles HTML pages, plain text responses, and PDF content.
"""

import re
import httpx
from io import BytesIO

# Optional PDF support
try:
    from pypdf import PdfReader
    _PDF_AVAILABLE = True
except ImportError:
    _PDF_AVAILABLE = False

# Optional HTML parsing
try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False


async def fetch_url(url: str) -> str:
    """Download a URL and return its text content, stripping HTML if needed."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; EzeeChatBot/1.0; +https://ezeechatbot.dev)"
        )
    }
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")

        if "application/pdf" in content_type:
            return _extract_pdf(response.content)

        if "text/html" in content_type:
            return _extract_html(response.text)

        # fallback — plain text, markdown, etc.
        return response.text


def _extract_html(raw_html: str) -> str:
    """Strip boilerplate (nav, footer, scripts) and return readable article text."""
    if not _BS4_AVAILABLE:
        # Crude regex fallback
        clean = re.sub(r"<[^>]+>", " ", raw_html)
        return re.sub(r"\s+", " ", clean).strip()

    soup = BeautifulSoup(raw_html, "html.parser")

    # Kill noise
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    # Prefer main article body if available
    main = soup.find("article") or soup.find("main") or soup.find("body")
    text = main.get_text(separator="\n") if main else soup.get_text(separator="\n")

    # Collapse excessive blank lines
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def _extract_pdf(raw_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    if not _PDF_AVAILABLE:
        raise RuntimeError(
            "PDF ingestion requires 'pypdf'. Install it with: pip install pypdf"
        )
    reader = PdfReader(BytesIO(raw_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def clean_text(raw: str) -> str:
    """Normalise whitespace and remove zero-width / control characters."""
    # Remove zero-width chars, non-printable control chars (keep newlines/tabs)
    raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200d\ufeff]", "", raw)
    # Collapse 3+ consecutive newlines into two
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    # Collapse runs of spaces/tabs into a single space
    raw = re.sub(r"[ \t]+", " ", raw)
    return raw.strip()
