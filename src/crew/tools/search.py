from crewai.tools import tool
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

@tool
def pinecone_search_tool(query: str, top_k: int = 2):
    """
    Search for relevant documents in Pinecone vector database.
    
    Args:
        query (str): The search query text
        top_k (int): Number of top results to return (default: 3)
        namespace (str): Pinecone namespace to search in (default: "__default__")
    
    Returns:
        Raw search results from Pinecone
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "kartavyaai"
    index = pc.Index(index_name)
    namespace = "__default__"
    results = index.search(
        namespace=namespace,
        query={
            "inputs": {"text": query},
            "top_k": top_k
        }
    )
    
    return results