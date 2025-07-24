from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bot_crew import BotCrew
import uvicorn
from typing import Optional
from tools.context import add_to_history
# Initialize FastAPI app
app = FastAPI(
    title="BotCrew API",
    description="API for querying KartavyaAI using CrewAI",
    version="1.0.0"
)

# Request model
class QueryRequest(BaseModel):
    query: str
    sender: str
    
class QueryResponse(BaseModel):
    query: str
    result: str
    status: str

# Initialize the crew once at startup
bot_crew = None

@app.on_event("startup")
async def startup_event():
    """Initialize BotCrew on startup"""
    global bot_crew
    try:
        print("Initializing BotCrew...")
        bot_crew = BotCrew()
        print("BotCrew initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize BotCrew: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "BotCrew API is running", "status": "healthy"}

@app.post("/query", response_model=QueryResponse)
async def query_bot_crew(request: QueryRequest):
    """
    Query the BotCrew with a user question, and store the interaction in memory.
    """
    global bot_crew

    if bot_crew is None:
        raise HTTPException(status_code=500, detail="BotCrew not initialized")

    try:
        # Prepare inputs
        inputs = {
            "query": request.query,
            "user_id":request.sender
        }

        print(f"Processing query: '{request.query}'")

        # Execute the Crew
        result = bot_crew.crew().kickoff(inputs=inputs)

        # Save to memory as a user-assistant message pair
        add_to_history([
            { "role": "user", "content": request.query },
            { "role": "assistant", "content": str(result) }
        ], user_id=request.sender)

        return QueryResponse(
            query=request.query,
            result=str(result),
            status="success"
        )

    except Exception as e:
        print(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Extended health check"""
    global bot_crew
    return {
        "status": "healthy" if bot_crew is not None else "unhealthy",
        "bot_crew_initialized": bot_crew is not None,
        "message": "BotCrew API is running"
    }

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "main:app",  # Replace "main" with your filename if different
        host="0.0.0.0",
        port=8000,
        reload=True
    )