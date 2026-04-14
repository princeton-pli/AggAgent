import asyncio
from typing import Optional, Union

from qwen_agent.tools.base import BaseTool, register_tool

from tools.tool_search_bcp import SearchBCP

@register_tool("get_document", allow_overwrite=True)
class GetDocumentBCP(BaseTool):
    name = "get_document"
    description = "Retrieve a full document by its docid."
    parameters = {
        "type": "object",
        "properties": {
            "docid": {
                "type": "string",
                "description": "Document ID to retrieve",
            }
        },
        "required": ["docid"],
    }
    
    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
    
    async def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            docid = params["docid"]
        except:
            return "[get_document] Invalid request format: Input must be a JSON object containing 'docid' field"

        try:
            searcher = await SearchBCP.get_searcher()
            result = await asyncio.to_thread(searcher.get_document, docid)
            if result is None:
                return f"[get_document] Document with docid '{docid}' not found"
        except Exception:
            return "[get_document] server error"

        return result

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
