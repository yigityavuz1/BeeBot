import json
import logging
from typing import List, Dict, Any, Literal
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END, StateGraph

from agent_state import State, QueryResponse
from prompts import CONTEXT_ANALYSIS_PROMPT, SYSTEM_PROMPT, CONTEXT_PROMPT, get_response_messages
from tools import setup_web_search, setup_double_retriever
from logger import logger

class EnhancedRAGSystem:
    """Enhanced RAG system that combines context sufficiency checking, web search,
    and double vector search with different alpha values."""
    
    def __init__(self, openai_api_key):
        """Initialize the RAG system with OpenAI API key."""
        self.openai_api_key = openai_api_key
        logger.info("Initializing EnhancedRAGSystem")
        self.llm = ChatOpenAI(
            model="gpt-4o", 
            temperature=0.2,
            api_key=openai_api_key,
        )
        self.json_llm = self.llm.bind(response_format={"type": "json_object"})
        self.memory = []
        self.graph = None
        logger.info("EnhancedRAGSystem initialized successfully")
    
    def create_graph(self, retriever):
        """Create the LangGraph workflow."""
        logger.info("Creating LangGraph workflow")
        # Set up tools and parsers
        double_retriever = setup_double_retriever(retriever)
        web_search = setup_web_search(self.openai_api_key)
        json_parser = JsonOutputParser(pydantic_object=QueryResponse)
        
        async def retrieve_context(state: State) -> State:
            """Retrieve context from the vector database."""
            query = state["query"]
            logger.info(f"Retrieving context for query: '{query}'")
            context = await double_retriever(query)
            state["retrieval_context"] = context
            logger.info(f"Retrieved {len(context)} context documents")
            return state
        
        async def analyze_context(state: State) -> State:
            """Analyze if the retrieved context is sufficient."""
            query = state["query"]
            context = state["retrieval_context"]
            
            # If no context is retrieved, definitely need web search
            if not context or len(context) == 0:
                logger.info(f"No context retrieved for query: '{query}'. Triggering web search.")
                state["needs_web_search"] = True
                return state
            
            # Log the amount of context retrieved
            logger.info(f"Retrieved {len(context)} context documents for query: '{query}'")
            
            # Check if we have sufficient content length in the retrieved documents
            total_content_length = sum(len(doc.page_content) for doc in context)
            if total_content_length < 200:  # If very little context was found
                logger.info(f"Retrieved context is too short ({total_content_length} chars). Triggering web search.")
                state["needs_web_search"] = True
                return state
            
            # Create a context analysis prompt
            chain = CONTEXT_ANALYSIS_PROMPT | self.json_llm | JsonOutputParser()
            
            try:
                logger.info("Invoking LLM for context sufficiency analysis")
                context_text = "\n\n".join([doc.page_content for doc in context])
                logger.info(f"Context analysis input length: {len(context_text)} chars")
                
                result = await chain.ainvoke({
                    "query": query,
                    "context": context_text
                })
                
                logger.info(f"Context analysis result: {result}")
                
                # Check if we got the expected format
                if not isinstance(result, dict) or 'sufficient' not in result:
                    logger.warning(f"Unexpected result format from context analysis: {result}")
                    # Default to using existing context if result format is unexpected
                    state["needs_web_search"] = False
                    return state
                
                state["needs_web_search"] = not result['sufficient']
                logger.info(f"Context sufficiency analysis: {result['sufficient']}. Web search needed: {state['needs_web_search']}")
                
            except Exception as e:
                logger.error(f"Error in context analysis: {str(e)}. Defaulting to using existing context.")
                # If there's an error in context analysis, default to using the existing context
                state["needs_web_search"] = False
            
            return state
        
        # Define the conditional routing function
        def decide_next_step(state: State) -> Literal["web_search", "generate_answer"]:
            """Decide whether to perform web search or go directly to generating answers."""
            if state["needs_web_search"]:
                logger.info("Router decision: Perform web search")
                return "web_search"
            else:
                logger.info("Router decision: Generate answer directly")
                return "generate_answer"
        
        async def perform_web_search(state: State) -> State:
            """Perform web search if needed using OpenAI's web search tool."""
            query = state["query"]
            logger.info(f"Performing OpenAI web search for query: '{query}'")
            
            try:
                # The search function is synchronous, so we don't need to await it
                search_results = web_search(query)
                state["web_search_results"] = search_results
                
                # Log detailed information about search results
                if search_results:
                    logger.info(f"OpenAI web search returned {len(search_results)} results")
                    for i, doc in enumerate(search_results):
                        logger.info(f"Search result {i+1} length: {len(doc.page_content)} chars")
                else:
                    logger.warning("OpenAI web search returned no results")
            except Exception as e:
                logger.error(f"Error during web search: {str(e)}")
                # Set empty results if there's an error
                state["web_search_results"] = []
            
            return state
        
        async def generate_answer(state: State) -> State:
            """Generate the final answer."""
            query = state["query"]
            memory = state["memory"]
            
            logger.info(f"Generating answer for query: '{query}'")
            
            documents = []
            source_type = "vector_db"
            source_details = {}
            
            # Collect strict and relaxed documents separately for formatting
            strict_docs = []
            relaxed_docs = []
            
            if state["retrieval_context"]:
                documents.extend(state["retrieval_context"])
                logger.info(f"Using {len(state['retrieval_context'])} documents from vector DB")
                
                # Separate documents based on search type
                for doc in state["retrieval_context"]:
                    if doc.metadata.get("search_type", "") == "strict (alpha=0.75)":
                        strict_docs.append(doc)
                    elif doc.metadata.get("search_type", "") == "relaxed (alpha=0.5)":
                        relaxed_docs.append(doc)
                
                logger.info(f"Using {len(strict_docs)} strict documents and {len(relaxed_docs)} relaxed documents")
            
            if state["needs_web_search"] and state["web_search_results"]:
                documents.extend(state["web_search_results"])
                source_type = "web_search" if not state["retrieval_context"] else "vector_db_and_web_search"
                logger.info(f"Using {len(state['web_search_results'])} documents from web search")
                
                if state["web_search_results"] and len(state["web_search_results"]) > 0:
                    search_content = state["web_search_results"][0].page_content
                    if "link:" in search_content:
                        link_start = search_content.find("link:") + 6
                        link_end = search_content.find(",", link_start) if "," in search_content[link_start:] else len(search_content)
                        url = search_content[link_start:link_end].strip()
                        source_details["webpage_url"] = url
                        logger.info(f"Extracted source URL from web search: {url}")
            
            if not documents:
                logger.warning("No documents found to answer the query")
                state["response"] = {
                    "answer": "Sorunuzu yanıtlamak için ilgili bilgi bulamadım.",
                    "source": "none",
                    "confidence": 0.0
                }
                return state
            
            chat_history = []
            for message in memory:
                if message["role"] == "human":
                    chat_history.append(HumanMessage(content=message["content"]))
                elif message["role"] == "ai":
                    chat_history.append(AIMessage(content=message["content"]))
            
            # Format contexts separately
            strict_content = "\n".join([doc.page_content for doc in strict_docs])
            relaxed_content = "\n".join([doc.page_content for doc in relaxed_docs])
            
            # Format Web Search Results if any
            web_content = ""
            if state["needs_web_search"] and state["web_search_results"]:
                web_content = "\n".join([doc.page_content for doc in state["web_search_results"]])
            
            # Create the messages for response generation
            messages = get_response_messages(
                SYSTEM_PROMPT, 
                CONTEXT_PROMPT, 
                source_type, 
                strict_content, 
                relaxed_content,
                web_content, 
                query
            )
            
            # Generate response
            logger.info("Invoking LLM for final answer generation")
            response = await self.json_llm.ainvoke(messages)
            logger.info(f"LLM response received, content length: {len(response.content)}")
            
            try:
                result = json_parser.parse(response.content)
                logger.info("Successfully parsed LLM response as JSON")
            except Exception as e:
                logger.error(f"Error parsing LLM response: {str(e)}")
                # Fallback if parsing fails
                result = {
                    "answer": response.content,
                    "source": source_type,
                    "confidence": 0.5
                }
                logger.info("Using fallback response due to parsing error")
            
            state["response"] = result
            return state
        
        # Create the workflow with conditional branching
        workflow = StateGraph(State)
        workflow.add_node("retrieve_context", retrieve_context)
        workflow.add_node("analyze_context", analyze_context)
        workflow.add_node("web_search", perform_web_search)
        workflow.add_node("generate_answer", generate_answer)
        
        # Set the entry point
        workflow.set_entry_point("retrieve_context")
        
        # Add conditional edges
        workflow.add_edge("retrieve_context", "analyze_context")
        workflow.add_conditional_edges(
            "analyze_context",
            decide_next_step,
            {
                "web_search": "web_search",
                "generate_answer": "generate_answer"
            }
        )
        workflow.add_edge("web_search", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        self.graph = workflow.compile()
        logger.info("LangGraph workflow created successfully")
        return self.graph
    
    async def process_query(self, query: str):
        """Process a user query through the RAG workflow."""
        if not self.graph:
            logger.error("Graph not created. Call create_graph first.")
            raise ValueError("Graph not created. Call create_graph first.")
            
        logger.info(f"Processing query: '{query}'")
        self.memory.append({"role": "human", "content": query})
        
        state = {
            "query": query,
            "retrieval_context": None,
            "web_search_results": None,
            "memory": self.memory.copy(),
            "response": None,
            "needs_web_search": False
        }
        
        try:
            logger.info("Invoking RAG graph workflow")
            result = await self.graph.ainvoke(state)
            logger.info("RAG graph workflow completed")
            
            if result["response"]:
                response_content = json.dumps(result["response"])
                logger.info(f"Response generated successfully, length: {len(result['response']['answer'])} chars")
                self.memory.append({"role": "ai", "content": response_content})
                return result["response"]
            else:
                logger.warning("No response generated by the workflow")
                fallback_response = {
                    "answer": "İsteğinizi işlerken bir hata oluştu.",
                    "source": "error",
                    "confidence": 0.0
                }
                self.memory.append({"role": "ai", "content": json.dumps(fallback_response)})
                return fallback_response
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            fallback_response = {
                "answer": f"İsteğinizi işlerken bir hata oluştu: {str(e)}",
                "source": "error",
                "confidence": 0.0
            }
            self.memory.append({"role": "ai", "content": json.dumps(fallback_response)})
            return fallback_response


async def extended_process_query(query, rag_system):
    """
    Extends the rag_system.process_query function to capture 
    and return the context information too.
    """
    logger.info(f"Extended query processing for '{query}'")
    # Create a custom state to track through the workflow
    state = {
        "query": query,
        "retrieval_context": None,
        "web_search_results": None,
        "memory": rag_system.memory.copy(),
        "response": None,
        "needs_web_search": False
    }
    
    try:
        # Process the query through the RAG system
        logger.info("Invoking RAG graph workflow for extended processing")
        result = await rag_system.graph.ainvoke(state)
        logger.info("Extended RAG graph workflow completed")
        response = result["response"]
        
        # Update the memory in the rag system
        rag_system.memory.append({"role": "human", "content": query})
        if response:
            rag_system.memory.append({"role": "ai", "content": json.dumps(response)})
        
        # Capture context for display
        context_info = {
            "strict_docs": [],
            "relaxed_docs": [],
            "web_results": [],
            "strict_metadata": [],
            "relaxed_metadata": []
        }
        
        # Extract documents by type
        if result.get("retrieval_context"):
            for doc in result["retrieval_context"]:
                if doc.metadata.get("search_type") == "strict (alpha=0.75)":
                    context_info["strict_docs"].append(doc.page_content)
                    context_info["strict_metadata"].append(doc.metadata)
                elif doc.metadata.get("search_type") == "relaxed (alpha=0.5)":
                    context_info["relaxed_docs"].append(doc.page_content)
                    context_info["relaxed_metadata"].append(doc.metadata)
        
        if result.get("web_search_results"):
            for doc in result["web_search_results"]:
                context_info["web_results"].append(doc.page_content)
        
        logger.info(f"Extended processing complete. Context info collected: {len(context_info['strict_docs'])} strict docs, " +
                   f"{len(context_info['relaxed_docs'])} relaxed docs, {len(context_info['web_results'])} web results")
        
        return response, context_info
        
    except Exception as e:
        logger.error(f"Error in extended query processing: {str(e)}")
        fallback_response = {
            "answer": f"İsteğinizi işlerken bir hata oluştu: {str(e)}",
            "source": "error",
            "confidence": 0.0
        }
        return fallback_response, None