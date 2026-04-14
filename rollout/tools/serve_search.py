import json
import os
import time
import asyncio
import logging
from functools import lru_cache
from typing import List, Dict, Optional, Tuple
import http.client
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
from rouge_score import rouge_scorer

from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai import CrawlerRunConfig, AsyncWebCrawler, BrowserConfig

# python serve_search.py --host 0.0.0.0 --port 8765 --workers 3

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Configuration
SERPER_KEY = os.environ.get('SERPER_KEY_ID')
SEARCH_CACHE_SIZE = int(os.environ.get('SEARCH_CACHE_SIZE', 8192))
VISIT_CACHE_SIZE = int(os.environ.get('VISIT_CACHE_SIZE', 8192))
VISIT_CONTENT_CACHE_SIZE = int(os.environ.get('VISIT_CONTENT_CACHE_SIZE', VISIT_CACHE_SIZE))
VISIT_SNIPPET_CACHE_SIZE = int(os.environ.get('VISIT_SNIPPET_CACHE_SIZE', VISIT_CACHE_SIZE))

# HTTP headers for web requests
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}

# Initialize ROUGE scorer for snippet finding
rouge_scorer_instance = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# Initialize FastAPI app
app = FastAPI(title="Search & Visit Service", version="1.0.0")

# Request/Response Models
class SearchRequest(BaseModel):
    query: str

class VisitRequest(BaseModel):
    url: str
    goal: str

class SearchResponse(BaseModel):
    query: str
    results: str
    cached: bool = False

class VisitResponse(BaseModel):
    url: str
    content: str
    cached: bool = False


def contains_chinese_basic(text: str) -> bool:
    """Check if text contains Chinese characters."""
    return any('\u4E00' <= char <= '\u9FFF' for char in text)


# Crawl4AI-based scraping functions
def detect_content_type(url: str) -> str:
    """Detect if URL points to PDF or HTML content."""
    parsed_url = urlparse(url)
    if parsed_url.path.lower().endswith('.pdf'):
        return 'pdf'

    try:
        # Use GET with stream to check first bytes (HEAD is unreliable for some servers)
        response = requests.get(url, headers=HEADERS, timeout=(3, 10), stream=True)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '')

        # Check Content-Type header
        if "pdf" in content_type:
            response.close()
            return "pdf"

        # Check magic bytes for PDF (%PDF)
        first_bytes = response.raw.read(5)
        response.close()
        if first_bytes.startswith(b'%PDF'):
            return "pdf"

        return "html"
    except Exception:
        return "html"


async def scrape_pdf(url: str) -> Tuple[bool, str, str]:
    """Scrape PDF content using PyMuPDF."""
    import fitz
    try:
        response = requests.get(url, headers=HEADERS, timeout=(3, 30))
        response.raise_for_status()
        with fitz.open(stream=response.content, filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return True, text, text
    except Exception as e:
        return False, str(e), ""


async def scrape_html(url: str) -> Tuple[bool, str, str]:
    """Scrape HTML content using crawl4ai."""
    prune_filter = PruningContentFilter(threshold=0.4, threshold_type="dynamic", min_word_threshold=3)
    md_generator = DefaultMarkdownGenerator(content_filter=prune_filter, options={"ignore_links": False})
    browser_config = BrowserConfig(
        headless=True, verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox", "--disable-extensions"]
    )
    crawler_config = CrawlerRunConfig(markdown_generator=md_generator, page_timeout=15000, verbose=False)

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await asyncio.wait_for(crawler.arun(url=url, config=crawler_config), timeout=30)

        if not result.success:
            return False, f"Failed to scrape the page: {result.error_message}", ""

        if len(result.markdown.raw_markdown.strip()) == 0:
            return False, "Failed to scrape the page: returned empty content.", ""

        fit_markdown = result.markdown.fit_markdown
        raw_markdown = result.markdown.raw_markdown

        return True, fit_markdown, raw_markdown
    except asyncio.TimeoutError:
        return False, "Timeout while scraping the page.", ""
    except Exception as e:
        return False, str(e), ""


def find_snippet(texts: List[str], query: str, num_characters: int = 10000, scoring_func: str = "rouge") -> str:
    """
    Find the most relevant snippet from texts based on the query.
    Uses ROUGE or BM25 scoring to find the best matching chunk.
    """
    positions = []
    start = 0
    best_recall = 0
    best_idx = 0

    if scoring_func == 'bm25':
        import bm25s
        import Stemmer
        stemmer = Stemmer.Stemmer('english')
        corpus_tokens = bm25s.tokenize(texts, stopwords='en', stemmer=stemmer)
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        query_tokens = bm25s.tokenize(query, stopwords='en', stemmer=stemmer)
        results, scores = retriever.retrieve(query_tokens, k=1)
        best_idx = int(results[0, 0])

    for i, text in enumerate(texts):
        if scoring_func == "rouge":
            score = rouge_scorer_instance.score(target=query, prediction=text)
            recall = score['rougeL'].recall
            if recall > best_recall:
                best_recall = recall
                best_idx = i
        positions.append((start, start + len(text)))
        start += len(text) + 1

    best_len = len(texts[best_idx])
    num_characters = num_characters - best_len
    final_text = []
    for i, pos in enumerate(positions):
        if (pos[0] >= positions[best_idx][0] - num_characters / 2 and
            pos[1] <= positions[best_idx][1] + num_characters / 2) or i == best_idx:
            final_text.append(texts[i])

    return "\n".join(final_text)


def get_content_sync(url: str) -> Tuple[bool, str, str]:
    """Synchronous wrapper to get raw content from URL."""
    try:
        content_type = detect_content_type(url)
        if content_type == "pdf":
            result = asyncio.run(scrape_pdf(url))
        else:
            result = asyncio.run(scrape_html(url))
        return result
    except Exception as e:
        return False, str(e), ""


def find_snippet_in_content(content: str, query: str, num_characters: int = 10000,
                            scoring_func: str = "rouge", chunking_func: str = "newline") -> Optional[str]:
    """Find relevant snippet in content based on query."""
    if not query:
        return content[:num_characters]

    if chunking_func == "newline":
        content_lines = content.split("\n")
        content_lines = [line for line in content_lines if line.strip()]
    elif "words" in chunking_func:
        num_words = int(chunking_func.split("_")[1])
        content_lines = content.split(" ")
        content_lines = [line for line in content_lines if line.strip()]
        content_lines = [content_lines[i:i + num_words] for i in range(0, len(content_lines), num_words)]
        content_lines = [" ".join(line) for line in content_lines]
    else:
        content_lines = content.split("\n")
        content_lines = [line for line in content_lines if line.strip()]

    if len(content_lines) == 0:
        return None

    return find_snippet(content_lines, query, num_characters, scoring_func)


def _visit_url_with_crawl4ai_uncached(
    url: str,
    goal: str,
    content_length: int = 10000,
    scoring_func: str = "rouge",
    chunking_func: str = "newline",
) -> str:
    success, content_or_error, raw_content = get_content_sync(url)
    if not success:
        return f"Failed to visit the url {url}.\nError: {content_or_error}"

    final_content = find_snippet_in_content(content_or_error, goal, content_length, scoring_func, chunking_func)
    if final_content is None:
        return f"Failed to extract content from {url}"

    return f"Successfully visited the url {url}.\n<Content>\n{final_content}\n</Content>"


@lru_cache(maxsize=VISIT_CONTENT_CACHE_SIZE)
def _get_content_cached(url: str) -> Tuple[bool, str, str]:
    """Cache the expensive URL fetching/scraping by URL only."""
    return get_content_sync(url)


@lru_cache(maxsize=VISIT_SNIPPET_CACHE_SIZE)
def _find_snippet_cached(
    content: str,
    goal: str,
    content_length: int = 10000,
    scoring_func: str = "rouge",
    chunking_func: str = "newline",
) -> Optional[str]:
    """Cache snippet extraction separately so different goals can reuse fetched content."""
    return find_snippet_in_content(content, goal, content_length, scoring_func, chunking_func)


def visit_url_with_crawl4ai(url: str, goal: str, content_length: int = 10000,
                            scoring_func: str = "rouge", chunking_func: str = "newline",
                            use_cache: bool = True) -> str:
    """Main function to visit a URL and extract relevant content based on goal using crawl4ai."""
    if not use_cache:
        return _visit_url_with_crawl4ai_uncached(url, goal, content_length, scoring_func, chunking_func)

    success, content_or_error, raw_content = _get_content_cached(url)
    if not success:
        return f"Failed to visit the url {url}.\nError: {content_or_error}"

    final_content = _find_snippet_cached(content_or_error, goal, content_length, scoring_func, chunking_func)
    if final_content is None:
        return f"Failed to extract content from {url}"

    return f"Successfully visited the url {url}.\n<Content>\n{final_content}\n</Content>"


# Core Search Function
def _google_search_with_serp_uncached(query: str) -> str:
    """
    Perform Google search using Serper API.
    """
    conn = http.client.HTTPSConnection("google.serper.dev", timeout=30)

    if contains_chinese_basic(query):
        payload = json.dumps({
            "q": query,
            "location": "China",
            "gl": "cn",
            "hl": "zh-cn"
        })
    else:
        payload = json.dumps({
            "q": query,
            "location": "United States",
            "gl": "us",
            "hl": "en"
        })

    headers = {
        'X-API-KEY': SERPER_KEY,
        'Content-Type': 'application/json'
    }

    # Retry logic
    for i in range(5):
        try:
            conn.request("POST", "/search", payload, headers)
            res = conn.getresponse()
            break
        except Exception as e:
            logger.warning(f"Search attempt {i+1} failed: {e}")
            if i == 4:
                return "Google search timeout. Please try again later."
            time.sleep(0.5)
            continue

    data = res.read()
    results = json.loads(data.decode("utf-8"))

    try:
        if "organic" not in results:
            raise Exception(f"No results found for query: '{query}'")

        web_snippets = []
        idx = 0

        for page in results["organic"]:
            idx += 1
            date_published = ""
            if "date" in page:
                date_published = "\nDate published: " + page["date"]

            source = ""
            if "source" in page:
                source = "\nSource: " + page["source"]

            snippet = ""
            if "snippet" in page:
                snippet = "\n" + page["snippet"]

            redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
            redacted_version = redacted_version.replace("Your browser can't play this video.", "")
            web_snippets.append(redacted_version)

        content = f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
        return content

    except Exception as e:
        logger.error(f"Search parsing error: {e}")
        return f"No results found for '{query}'. Try with a more general query."


@lru_cache(maxsize=SEARCH_CACHE_SIZE)
def _google_search_with_serp_cached(query: str) -> str:
    return _google_search_with_serp_uncached(query)


def google_search_with_serp(query: str, use_cache: bool = True) -> str:
    """
    Perform Google search using Serper API with optional in-memory LRU caching.
    """
    if use_cache:
        return _google_search_with_serp_cached(query)
    return _google_search_with_serp_uncached(query)

# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "Search & Visit Service",
        "endpoints": ["/search", "/visit", "/clear_cache"]
    }


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Perform a Google search and return formatted results.
    """
    try:
        search_hits_before = _google_search_with_serp_cached.cache_info().hits
        # Run blocking function in thread pool
        results = await asyncio.to_thread(google_search_with_serp, request.query)
        cached = _google_search_with_serp_cached.cache_info().hits > search_hits_before
        logger.info(f"Search request: {request.query} (cached={cached})")
        return SearchResponse(
            query=request.query,
            results=results,
            cached=cached
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/visit", response_model=VisitResponse)
async def visit(request: VisitRequest):
    """
    Visit a URL and extract relevant content based on goal using crawl4ai.
    """
    try:
        content_hits_before = _get_content_cached.cache_info().hits
        snippet_hits_before = _find_snippet_cached.cache_info().hits
        # Run blocking function in thread pool
        content = await asyncio.to_thread(
            visit_url_with_crawl4ai,
            request.url,
            request.goal,
            10000,  # content_length
            "rouge",  # scoring_func
            "newline"  # chunking_func
        )
        cached = (
            _get_content_cached.cache_info().hits > content_hits_before
            or _find_snippet_cached.cache_info().hits > snippet_hits_before
        )
        logger.info(f"Visit request: {request.url} (cached={cached})")
        return VisitResponse(
            url=request.url,
            content=content,
            cached=cached
        )
    except Exception as e:
        logger.error(f"Visit error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear_cache")
async def clear_cache(cache_type: Optional[str] = None):
    """
    Clear in-memory LRU caches. Specify cache_type ('search' or 'visit') or clear all.
    """
    try:
        cleared = []
        if cache_type in (None, "search"):
            _google_search_with_serp_cached.cache_clear()
            cleared.append("search")
        if cache_type in (None, "visit"):
            _get_content_cached.cache_clear()
            _find_snippet_cached.cache_clear()
            cleared.append("visit")
        if cache_type not in (None, "search", "visit"):
            raise HTTPException(status_code=400, detail="cache_type must be one of: search, visit")

        return {
            "status": "success",
            "cleared": cleared,
            "type": cache_type or "all",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Search & Visit Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8006, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")

    args = parser.parse_args()

    logger.info(f"Starting Search & Visit Service on {args.host}:{args.port}")
    logger.info(f"Search cache size: {SEARCH_CACHE_SIZE}")
    logger.info(f"Visit content cache size: {VISIT_CONTENT_CACHE_SIZE}")
    logger.info(f"Visit snippet cache size: {VISIT_SNIPPET_CACHE_SIZE}")
    logger.info(f"Workers: {args.workers}")

    if args.workers > 1:
        # Multiple workers require import string - must run from tools directory
        uvicorn.run(
            "serve_search:app",
            host=args.host,
            port=args.port,
            workers=args.workers
        )
    else:
        # Single worker can use app object directly
        uvicorn.run(
            app,
            host=args.host,
            port=args.port
        )
