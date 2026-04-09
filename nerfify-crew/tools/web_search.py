"""
Web search and fetch tools for CrewAI agents.

Provides the same WebSearch and WebFetch capabilities that the Claude SDK version uses.
Uses the `ddgs` package (formerly `duckduckgo-search`) for web search.
"""
from __future__ import annotations

import httpx
from crewai.tools import BaseTool
from pydantic import Field

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS


class WebSearchTool(BaseTool):
    """Search the web using DuckDuckGo and return relevant results."""

    name: str = "web_search"
    description: str = (
        "Search the web for information. Use this to find research papers, "
        "NeRFStudio documentation, PyTorch patterns, GitHub repos, and solutions "
        "to error messages. Returns a list of search results with titles, URLs, and snippets."
    )
    max_results: int = Field(default=8, description="Maximum number of results to return")

    def _run(self, query: str) -> str:
        try:
            results = DDGS().text(query, max_results=self.max_results)
            if not results:
                return f"No results found for: {query}"
            output_parts = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "No title")
                url = r.get("href", r.get("link", ""))
                snippet = r.get("body", r.get("snippet", ""))
                output_parts.append(f"{i}. **{title}**\n   URL: {url}\n   {snippet}")
            return "\n\n".join(output_parts)
        except Exception as e:
            return f"Search error: {e}"


class WebFetchTool(BaseTool):
    """Fetch the content of a web page and return its text."""

    name: str = "web_fetch"
    description: str = (
        "Fetch the content of a URL and return the text. Use this to read "
        "research papers on arXiv, documentation pages, GitHub READMEs, and "
        "other web pages. Provide the full URL."
    )
    max_chars: int = Field(default=15000, description="Maximum characters to return")

    def _run(self, url: str) -> str:
        try:
            with httpx.Client(follow_redirects=True, timeout=30.0) as client:
                resp = client.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (compatible; NerfifyBot/1.0)"
                    },
                )
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "")

                if "text/html" in content_type:
                    # Basic HTML to text extraction
                    text = self._html_to_text(resp.text)
                else:
                    text = resp.text

                if len(text) > self.max_chars:
                    text = text[: self.max_chars] + "\n... [truncated]"
                return text
        except Exception as e:
            return f"Fetch error for {url}: {e}"

    @staticmethod
    def _html_to_text(html: str) -> str:
        """Simple HTML to text conversion."""
        import re

        # Remove script and style tags
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Decode common entities
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Re-add some line breaks for readability
        text = re.sub(r"\s{3,}", "\n\n", text)
        return text
