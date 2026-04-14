import asyncio
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union

from transformers import AutoTokenizer
from qwen_agent.tools.base import BaseTool, register_tool

from searchers import SearcherType

@register_tool("search", allow_overwrite=True)
class SearchBCP(BaseTool):
    name = "search"
    description = "Perform a search on a knowledge source. Returns top-5 hits with docid, score, and snippet. The snippet contains the document's contents (may be truncated based on token limits)."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string",
            }
        },
        "required": ["query"],
    }
    
    _searcher = None
    _searcher_lock = None
    _snippet_tokenizer = None

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
    
    @classmethod
    def _get_env_int(cls, key: str, default: int) -> int:
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    @classmethod
    def _get_env_bool(cls, key: str, default: bool = False) -> bool:
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in {"1", "true", "yes", "y", "on"}

    @classmethod
    def _build_searcher_from_env(cls):
        searcher_type = os.getenv("SEARCHER_TYPE", "faiss")
        searcher_cls = SearcherType.get_searcher_class(searcher_type)

        args = SimpleNamespace(
            index_path=os.getenv("INDEX_PATH"),
            model_name=os.getenv("SEARCH_MODEL_NAME"),
            normalize=cls._get_env_bool("SEARCH_NORMALIZE", True),
            pooling=os.getenv("SEARCH_POOLING", "eos"),
            torch_dtype=os.getenv("SEARCH_TORCH_DTYPE", "float16"),
            dataset_name=os.getenv("SEARCH_DATASET_NAME"),
            task_prefix=os.getenv(
                "SEARCH_TASK_PREFIX",
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
            ),
            max_length=cls._get_env_int("SEARCH_MAX_LENGTH", 8192),
        )

        if not args.index_path:
            raise ValueError("INDEX_PATH is required for local searcher (e.g. /path/to/corpus.shard*.pkl)")

        return searcher_cls(args)

    @classmethod
    async def get_searcher(cls):
        if cls._searcher is not None:
            return cls._searcher

        # Lazy initialization of lock (must be done in async context)
        if cls._searcher_lock is None:
            cls._searcher_lock = asyncio.Lock()

        async with cls._searcher_lock:
            if cls._searcher is None:
                cls._searcher = await asyncio.to_thread(cls._build_searcher_from_env)
        return cls._searcher

    @classmethod
    async def get_snippet_tokenizer(cls):
        snippet_max_tokens = cls._get_env_int("SNIPPET_MAX_TOKENS", 512)
        if snippet_max_tokens <= 0:
            return None
        if cls._snippet_tokenizer is None:
            tokenizer_path = os.getenv("SNIPPET_TOKENIZER_PATH")
            cls._snippet_tokenizer = await asyncio.to_thread(
                AutoTokenizer.from_pretrained, tokenizer_path
            )
        return cls._snippet_tokenizer
    
    async def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            query = params["query"]
        except:
            return "[search] Invalid request format: Input must be a JSON object containing 'query' field"

        try:
            searcher = await self.get_searcher()
            k = self._get_env_int("SEARCH_K", 5)
            candidates = await asyncio.to_thread(searcher.search, query, k)

            snippet_max_tokens = self._get_env_int("SNIPPET_MAX_TOKENS", 512)
            if snippet_max_tokens > 0:
                tokenizer = await self.get_snippet_tokenizer()
            else:
                tokenizer = None

            if snippet_max_tokens > 0 and tokenizer:
                for cand in candidates:
                    text = cand["text"]
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                    if len(tokens) > snippet_max_tokens:
                        truncated_tokens = tokens[:snippet_max_tokens]
                        cand["snippet"] = tokenizer.decode(
                            truncated_tokens, skip_special_tokens=True
                        )
                    else:
                        cand["snippet"] = text
            else:
                for cand in candidates:
                    cand["snippet"] = cand["text"]

            results: List[Dict[str, Any]] = []
            for cand in candidates:
                if cand.get("score") is None:
                    results.append({"docid": cand["docid"], "snippet": cand["snippet"]})
                else:
                    results.append(
                        {
                            "docid": cand["docid"],
                            "score": cand["score"],
                            "snippet": cand["snippet"],
                        }
                    )
        except Exception as e:
            return "[search] server error"
        return results

    def get_tool_definitions(self):
        parameters = self.parameters.copy()
        parameters["additionalProperties"] = False
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }
