import json
import logging
from logger import logger
from typing import List
from langchain.docstore.document import Document
from openai import OpenAI
import re
from prompts import WEB_SEARCH_PROMPT

def setup_web_search(openai_api_key):
    """Setup OpenAI web search tool."""
    client = OpenAI(api_key=openai_api_key)
    
    def search(query: str) -> List[Document]:
        final_query = f"İstanbul Teknik Üniversitesi {query}"
        logger.info(f"Performing OpenAI web search with query: '{final_query}'")
        
        try:
            # Format the search prompt with the query
            formatted_prompt = WEB_SEARCH_PROMPT.format(query=final_query)
            
            # Use OpenAI's web search tool
            logger.info("Calling OpenAI with web search tool enabled")
            response = client.responses.create(
                model="gpt-4o",
                tools=[{"type": "web_search_preview"}],
                input=formatted_prompt
            )
            
            # Extract the response text
            search_result_text = response.output_text
            logger.info(f"Received web search results from OpenAI (length: {len(search_result_text)})")
            
            # Parse the structured results
            documents = []
            
            # Extract results using regex
            result_pattern = r'\[SONUÇ\](.*?)\[/SONUÇ\]'
            results = re.findall(result_pattern, search_result_text, re.DOTALL)
            
            for result in results:
                # Create a document for each result
                documents.append(Document(
                    page_content=result.strip(),
                    metadata={"source": "openai_web_search"}
                ))
            
            # If no structured results were found, use the entire text as one document
            if not documents:
                logger.info("No structured results found in response, using entire response as one document")
                documents.append(Document(
                    page_content=search_result_text,
                    metadata={"source": "openai_web_search"}
                ))
            
            logger.info(f"Created {len(documents)} documents from OpenAI web search results")
            return documents
        
        except Exception as e:
            logger.error(f"Error in OpenAI web search: {str(e)}")
            # Return an empty list in case of an error
            return []
    
    return search

def setup_double_retriever(retriever):
    """Setup a retriever that performs two searches with different alpha values."""
    async def double_retriever(query: str) -> List[Document]:
        logger.info(f"Performing double retrieval with query: '{query}'")
        
        # Get documents with different alpha values
        logger.info("Retrieving documents with strict alpha (0.75)")
        docs_strict = await retriever(query, k=3, alpha=0.75)
        logger.info(f"Retrieved {len(docs_strict)} documents with strict alpha")
        
        logger.info("Retrieving documents with relaxed alpha (0.5)")
        docs_relaxed = await retriever(query, k=3, alpha=0.5)
        logger.info(f"Retrieved {len(docs_relaxed)} documents with relaxed alpha")
        
        # Add metadata to identify the source
        for doc in docs_strict:
            doc.metadata["search_type"] = "strict (alpha=0.75)"
        
        for doc in docs_relaxed:
            doc.metadata["search_type"] = "relaxed (alpha=0.5)"
        
        # Combine both sets of documents
        combined_docs = docs_strict + docs_relaxed
        logger.info(f"Combined retrieval returned {len(combined_docs)} total documents")
        return combined_docs
    
    return double_retriever

def format_context(context_info):
    """Format the context information for display."""
    formatted_context = ""
    
    if context_info["strict_docs"]:
        formatted_context += "### BAĞLAM - BELGE 1 (alpha=0.75)\n"
        for i, content in enumerate(context_info["strict_docs"]):
            formatted_context += f"#### Belge 1.{i+1}\n"
            formatted_context += content + "\n"
            if i < len(context_info["strict_metadata"]):
                formatted_context += f"**Metadata:** {json.dumps(context_info['strict_metadata'][i], ensure_ascii=False)}\n\n"
    
    if context_info["relaxed_docs"]:
        formatted_context += "### BAĞLAM - BELGE 2 (alpha=0.5)\n"
        for i, content in enumerate(context_info["relaxed_docs"]):
            formatted_context += f"#### Belge 2.{i+1}\n"
            formatted_context += content + "\n"
            if i < len(context_info["relaxed_metadata"]):
                formatted_context += f"**Metadata:** {json.dumps(context_info['relaxed_metadata'][i], ensure_ascii=False)}\n\n"
    
    if context_info["web_results"]:
        formatted_context += "### WEB ARAMA SONUÇLARI\n"
        for i, content in enumerate(context_info["web_results"]):
            formatted_context += f"#### Web Sonuç {i+1}\n"
            formatted_context += content + "\n\n"
    
    return formatted_context if formatted_context else "Bağlam bulunamadı."

def format_metadata(response):
    """Format the metadata for display."""
    metadata = {
        "Kaynak": response.get("source", "Bilinmiyor"),
        "Güven Skoru": f"{response.get('confidence', 0.0):.2f}"
    }
    
    return json.dumps(metadata, indent=2, ensure_ascii=False)

def run_async(loop, coro):
    """Run an async function in Streamlit without closing the event loop."""
    return loop.run_until_complete(coro)