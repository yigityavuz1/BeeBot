from typing import List, Dict, Any, Optional, TypedDict, Literal
from pydantic import BaseModel, Field
from langchain.docstore.document import Document

class QueryResponse(BaseModel):
    """Pydantic model for query response."""
    answer: str = Field(description="The answer to the user's query")
    source: str = Field(description="The source of the information (vector DB or web search)")
    confidence: float = Field(description="Confidence score from 0 to 1")

class State(TypedDict):
    """TypedDict for the state of the workflow."""
    query: str
    retrieval_context: Optional[List[Document]]
    web_search_results: Optional[List[Document]]
    memory: List[Dict[str, str]]
    response: Optional[Dict[str, Any]]
    needs_web_search: bool

class ContextInfo(TypedDict):
    """TypedDict for storing context information."""
    strict_docs: List[str]
    relaxed_docs: List[str]
    web_results: List[str]
    strict_metadata: List[Dict]
    relaxed_metadata: List[Dict]