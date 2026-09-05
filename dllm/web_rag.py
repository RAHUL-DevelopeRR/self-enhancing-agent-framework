from __future__ import annotations

import hashlib
import html
import ipaddress
from html.parser import HTMLParser
from typing import List
from urllib.parse import urlparse

import httpx

from .config import WebRagConfig
from .cntx import CntxStore
from .schemas import CntxEvent, EventType, Priority


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: List[str] = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden and data.strip():
            self.parts.append(data.strip())


class SearxngWebRag:
    """Optional web RAG: search, bounded fetch, provenance, then append to `.cntx`."""

    def __init__(self, config: WebRagConfig, store: CntxStore):
        self.config = config
        self.store = store

    def _allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return False
        hostname = (parsed.hostname or "").lower()
        if hostname == "localhost" or hostname.endswith(".localhost"):
            return False
        try:
            address = ipaddress.ip_address(hostname)
            if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved:
                return False
        except ValueError:
            pass
        if not self.config.allowed_domains:
            return True
        return any(hostname == domain or hostname.endswith(f".{domain}") for domain in self.config.allowed_domains)

    async def retrieve(self, query: str, task_id: str) -> List[str]:
        if not self.config.enabled or not self.config.searxng_url:
            return []
        async with httpx.AsyncClient(timeout=self.config.fetch_timeout_seconds, follow_redirects=True) as client:
            response = await client.get(
                f"{self.config.searxng_url.rstrip('/')}/search",
                params={"q": query, "format": "json"},
            )
            response.raise_for_status()
            candidates = [item for item in response.json().get("results", []) if self._allowed(item.get("url", ""))]
            event_ids = []
            for item in candidates[: self.config.result_limit]:
                url = item.get("url", "")
                try:
                    page = await client.get(url)
                    page.raise_for_status()
                    if not self._allowed(str(page.url)):
                        raise httpx.HTTPError("Redirected to a disallowed web target")
                    parser = _TextExtractor()
                    parser.feed(page.text[: self.config.max_document_chars * 4])
                    content = html.unescape("\n".join(parser.parts))[: self.config.max_document_chars]
                except httpx.HTTPError:
                    content = item.get("content", "")[: self.config.max_document_chars]
                event = self.store.append(CntxEvent(
                    type=EventType.WEB_EVIDENCE,
                    scope="web",
                    priority=Priority.P2,
                    source=f"web:{urlparse(url).hostname}",
                    task_id=task_id,
                    summary=item.get("title") or url,
                    payload={
                        "url": url,
                        "snippet": item.get("content", ""),
                        "content": content,
                        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    },
                    provenance=[url],
                ))
                event_ids.append(f"cntx:{event.id}")
            return event_ids
