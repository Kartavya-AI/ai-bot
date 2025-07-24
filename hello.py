from typing import List, Dict,Union
import json
import os
from dotenv import load_dotenv
from mem0 import MemoryClient
from crewai.tools import tool

# Load environment variables
load_dotenv()
client = MemoryClient(api_key="os.getenv('MEMORY_API_KEY')")

def get_from_memory(query: str) -> List[Dict[str, Union[str, list]]]:
    """
    Retrieve formatted memories from memory storage by query.

    Args:
        query (str): The search query string

    Returns:
        List[Dict]: A list of formatted memory entries
    """
    results = client.search(query, user_id="Sarthak")

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

# Example usage
formatted_memories = get_from_memory("what is user interested in?")
print(json.dumps(formatted_memories, indent=2))
