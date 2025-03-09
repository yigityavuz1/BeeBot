import json
from typing import List
from langchain.docstore.document import Document
from langchain_community.tools import DuckDuckGoSearchResults

def setup_web_search():
    """Setup DuckDuckGo web search tool."""
    search_tool = DuckDuckGoSearchResults(num_results=3)
    
    def search(query: str) -> List[Document]:
        final_query = f"İstanbul Teknik Üniversitesi {query}"
        search_results = search_tool.invoke(final_query)
        documents = []
        
        for result in search_results.split('\n'):
            if result.strip():
                documents.append(Document(page_content=result))
        
        return documents
    
    return search

def setup_double_retriever(retriever):
    """Setup a retriever that performs two searches with different alpha values."""
    async def double_retriever(query: str) -> List[Document]:
        # Get documents with different alpha values
        docs_strict = await retriever(query, k=3, alpha=0.75)
        docs_relaxed = await retriever(query, k=3, alpha=0.5)
        
        # Add metadata to identify the source
        for doc in docs_strict:
            doc.metadata["search_type"] = "strict (alpha=0.75)"
        
        for doc in docs_relaxed:
            doc.metadata["search_type"] = "relaxed (alpha=0.5)"
        
        # Combine both sets of documents
        combined_docs = docs_strict + docs_relaxed
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