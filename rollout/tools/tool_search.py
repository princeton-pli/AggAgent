"""
Client wrapper for Search tool that uses serve_search.py backend.
Maintains the same API as tool_search.py but with caching benefits.
"""

import json
import asyncio
from typing import List, Union, Optional
import requests
from qwen_agent.tools.base import BaseTool, register_tool
import os
from dotenv import load_dotenv

load_dotenv()

SEARCH_SERVER_URL = os.environ.get('SEARCH_SERVER_URL', 'http://localhost:8765')


@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    name = "search"
    description = "Performs a web search: supply a string 'query'; the tool retrieves the top 10 results for the query."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The query string for the search."
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        self.server_url = SEARCH_SERVER_URL

    def search_with_server(self, query: str) -> str:
        """Send search request to serve_search.py backend."""
        try:
            response = requests.post(
                f"{self.server_url}/search",
                json={"query": query},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            # if data.get('cached'):
            #     print(f"[Search] Cache hit for: {query}")

            return data['results']

        except requests.exceptions.ConnectionError:
            return "[Search] Error: Cannot connect to server."
        except requests.exceptions.Timeout:
            return "[Search] Error: Search request timed out."
        except Exception as e:
            return f"[Search] Error: {str(e)}"

    async def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            query = params["query"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"

        if isinstance(query, str):
            response = await asyncio.to_thread(self.search_with_server, query)
        else:
            # Handle list of queries
            assert isinstance(query, list)
            responses = []
            for q in query:
                res = await asyncio.to_thread(self.search_with_server, q)
                responses.append(res)
            response = "\n=======\n".join(responses)

        return response

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
