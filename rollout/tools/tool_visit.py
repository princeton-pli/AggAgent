"""
Client wrapper for Visit tool that uses serve_search.py backend.
Uses crawl4ai for scraping with ROUGE/BM25 for snippet finding.
"""

import asyncio
from typing import Union, Optional
import requests
import os
from qwen_agent.tools.base import BaseTool, register_tool
from dotenv import load_dotenv

load_dotenv()

SEARCH_SERVER_URL = os.environ.get('SEARCH_SERVER_URL', 'http://localhost:8765')


@register_tool('visit', allow_overwrite=True)
class Visit(BaseTool):
    name = 'visit'
    description = 'Visit a webpage and return the relevant content based on the goal.'
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL of the webpage to visit."
            },
            "goal": {
                "type": "string",
                "description": "The goal of the visit for the webpage."
            }
        },
        "required": ["url", "goal"]
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        self.server_url = SEARCH_SERVER_URL

    async def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            url = params["url"]
            goal = params["goal"]
        except:
            return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"

        response = await asyncio.to_thread(self.visit_with_server, url, goal)
        return response.strip()

    def visit_with_server(self, url: str, goal: str) -> str:
        """Send visit request to serve_search.py backend."""
        try:
            response = requests.post(
                f"{self.server_url}/visit",
                json={"url": url, "goal": goal},
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

            # if data.get('cached'):
            #     print(f"[Visit] Cache hit for: {url}")

            return data['content']

        except requests.exceptions.ConnectionError:
            return "[Visit] Error: Cannot connect to server."
        except requests.exceptions.Timeout:
            return "[Visit] Error: Visit request timed out."
        except Exception as e:
            return f"[Visit] Error: {str(e)}"

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
