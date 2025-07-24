from typing import Union,List, Dict
import json
from mem0 import MemoryClient
from crewai.tools import tool
client = MemoryClient(api_key= os.getenv("MEMORY_API_KEY"))

def add_to_history(content: Union[str, list, dict], user_id: str = "Sarthak") -> str:
    """
    Adds content or chat history to memory.

    Args:
        content (str | list | dict): A string (user message), or a list of messages like:
            [
                { "role": "user", "content": "Hi" },
                { "role": "assistant", "content": "Hello" }
            ]
        user_id (str): ID of the user to associate memory with.

    Returns:
        str: Success or error message.
    """
    try:
        if isinstance(content, str):
            messages = [{"role": "user", "content": content}]
        elif isinstance(content, list):
            for m in content:
                if not isinstance(m, dict) or 'role' not in m or 'content' not in m:
                    return "Invalid message format. Each item must be a dict with 'role' and 'content'."
            messages = content
        elif isinstance(content, dict):
            if 'role' in content and 'content' in content:
                messages = [content]
            else:
                return "Invalid dict format. It must include 'role' and 'content'."
        else:
            return "Unsupported input. Provide a string, dict, or list of message dicts."

        client.add(messages, user_id=user_id)
        return f"Successfully added to memory: {json.dumps(messages)}"
    except Exception as e:
        return f"Error adding to memory: {str(e)}"

@tool
def get_from_memory(request_body: str) -> Union[str, List[Dict[str, Union[str, list]]]]:
    """
    Retrieve and format memory entries based on a JSON string containing query and user_id.

    Args:
        request_body (str): A JSON string with "query" and "user_id".
            Example: '{"query": "user name", "user_id": "7838034911"}'

    Returns:
        Union[str, List[Dict]]: A list of formatted memory entries or a message if none found.
    """
    try:
        data = json.loads(request_body)
        query = data.get("query")
        user_id = data.get("user_id")
        
        if not query or not user_id:
            return "Both 'query' and 'user_id' must be provided in the request."

        results = client.search(query, user_id=user_id)

        if not results:
            return "No memory found for the given query."

        formatted = [
            {
                "id": r["id"],
                "memory": r["memory"],
                "categories": r.get("categories", []),
                "created_at": r["created_at"]
            }
            for r in results
        ]
        return formatted

    except json.JSONDecodeError:
        return "Invalid JSON input."