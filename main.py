import os
import sys
import time
from colorama import init, Fore, Style
from dotenv import load_dotenv
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Initialize colorama
init(autoreset=True)

# Load environment
load_dotenv()

# Import local components
try:
    from planner import PlanningAgent
    from search_engine import GraphRAGQueryEngine
    from guardrails import GuardrailViolation
except ImportError:
    # Handle path issues if running from another directory
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from planner import PlanningAgent
    from search_engine import GraphRAGQueryEngine
    from guardrails import GuardrailViolation

origins = [
    "http://localhost:5173",    # Common Vite/Vue/Svelte port
    "https://review-analyzer-d4x8.onrender.com" # Your production frontend domain
]

# FastAPI Instance
app = FastAPI(
    title="GraphRAG Adaptive Query Planning API",
    description="HTTP API for adaptive routing and querying of Microsoft GraphRAG using Gemini models.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # Allows specific domains
    # allow_origins=["*"],            # OR Use this wildcard ONLY for public APIs (Dangerous for private data)
    allow_credentials=True,           # Allows cookies and credentials
    allow_methods=["*"],              # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],              # Allows all request headers
)

# Global instances of agents
planner = PlanningAgent()
engine = GraphRAGQueryEngine()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    query_type: str
    reasoning: str
    response: str

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """
    POST API endpoint to submit a query. It will be planned/routed dynamically and
    executed against the Microsoft GraphRAG engine.
    """
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        print(f"\n[API] Received query: '{request.query}'")

        # Route query through planner
        plan = planner.plan_query(request.query)
        if not plan:
            raise HTTPException(status_code=500, detail="Failed to plan query")

        print(f"[API] Routed query as '{plan.query_type.upper()}' search. Reasoning: {plan.reasoning}")

        # Execute query using real GraphRAG engine
        result = engine.run_query(request.query, plan)
        return QueryResponse(
            query=request.query,
            query_type=plan.query_type,
            reasoning=plan.reasoning,
            response=result
        )
    except GuardrailViolation as e:
        print(f"[API Warning] Guardrail Violation: {e.detail}")
        raise HTTPException(status_code=400, detail=f"Guardrail violation: {e.detail}")
    except Exception as e:
        print(f"[API Error] Exception during query handling: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def print_banner():
    banner = f"""
{Fore.CYAN}{Style.BRIGHT}========================================================================
{Fore.MAGENTA}{Style.BRIGHT}             GRAPH RAG ADAPTIVE QUERY PLANNING AGENT
{Fore.CYAN}{Style.BRIGHT}========================================================================
{Fore.WHITE} Powered by: {Fore.GREEN}Gemini 3.5 Flash Lite / ChatGPT
{Fore.WHITE} Dataset:    {Fore.YELLOW}Project Aether Knowledge Graph (Geneva Labs)
{Fore.WHITE} Modes:      {Fore.CYAN}Global{Fore.WHITE}, {Fore.YELLOW}Local{Fore.WHITE}, {Fore.MAGENTA}Drift{Fore.WHITE}, {Fore.GREEN}Basic
========================================================================
    """
    print(banner)


if __name__ == "__main__":
    import uvicorn
    print_banner()
    print("[System] Starting FastAPI application server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
