import os
import glob
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import tiktoken
from dotenv import load_dotenv

class DataProcessor:
    """Class for processing text data and converting it to LangChain documents."""
    
    def __init__(self):
        """Initialize the data processor with environment variables for directories."""
        load_dotenv()
        self.raw_data_dir = os.getenv("RAW_DATA_DIR", "raw")
        self.pdf_text_dir = os.getenv("PDF_TEXT_DIR", "pdf_text")
        self.processed_data_dir = os.getenv("PROCESSED_DATA_DIR", "processed")
        
        # Initialize tokenizer for OpenAI models
        self.encoding = tiktoken.encoding_for_model("text-embedding-3-large")
        
        # Configure text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,    
            chunk_overlap=400,  
            separators=["\n\n", "\n", " ", ""]
        )
    
    def create_langchain_documents_raw(self) -> List[Document]:
        """Creates LangChain Document objects from .txt files in the raw data directory."""
        directory = self.raw_data_dir
        if not os.path.isdir(directory):
            print(f"Warning: {directory} is not a valid directory. Skipping raw document processing.")
            return []

        documents = []
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)

            if file.endswith(".txt") and os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": file[:-4],  # Remove .txt extension
                            "content_length": len(content)
                        }
                    )
                    documents.append(doc)
                    print(f"Processed raw file: {file}")

                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

        print(f"Processed {len(documents)} raw documents")
        return documents
    
    def parse_pdf_text_files(self) -> List[Document]:
        """Process text files from PDF conversions in the PDF text directory."""
        directory = self.pdf_text_dir
        all_docs = []
        
        if not os.path.isdir(directory):
            print(f"Warning: {directory} is not a valid directory. Skipping PDF text processing.")
            return []
            
        txt_file_paths = glob.glob(os.path.join(directory, "*.txt"))
        
        for file_path in txt_file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    full_text = f.read()

                # Add metadata about the source
                metadata = {
                    "source": os.path.basename(file_path),
                    "content_length": len(full_text),
                }

                # Create chunks
                chunks = self.text_splitter.split_text(full_text)

                # Create Document objects for each chunk
                for i, chunk in enumerate(chunks):
                    chunk_metadata = {
                        **metadata,
                        "chunk_index": f"{os.path.basename(file_path)}_{i}",
                    }
                    doc = Document(page_content=chunk, metadata=chunk_metadata)
                    all_docs.append(doc)
                
                print(f"Processed PDF text file: {os.path.basename(file_path)} into {len(chunks)} chunks")
            
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        print(f"Processed {len(all_docs)} PDF text chunks")
        return all_docs
    
    def create_langchain_documents_results(self) -> List[Document]:
        """Creates LangChain Document objects from .txt files in the processed data directory."""
        directory = self.processed_data_dir
        if not os.path.isdir(directory):
            print(f"Warning: {directory} is not a valid directory. Skipping results processing.")
            return []

        documents = []
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)

            if file.endswith(".txt") and os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": file[:-4],  # Remove .txt extension
                            "content_length": len(content)
                        }
                    )
                    documents.append(doc)
                    print(f"Processed results file: {file}")

                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

        print(f"Processed {len(documents)} result documents")
        return documents
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the OpenAI tokenizer"""
        return len(self.encoding.encode(text))
    
    def split_text(self, text: str, overlap_percentage: int) -> tuple:
        """Split text in half with specified overlap"""
        if not text:
            return [], []
            
        # Calculate midpoint
        midpoint = len(text) // 2
        
        # Calculate overlap size in characters
        overlap_chars = int(len(text) * (overlap_percentage / 100))
        
        # Create the two chunks with overlap
        first_chunk = text[:midpoint + overlap_chars]
        second_chunk = text[midpoint - overlap_chars:]
        
        return first_chunk, second_chunk
    
    def chunk_documents_by_tokens(self, documents: List[Document], max_tokens: int = 8000, 
                                  overlap_percentage: int = 20) -> List[Document]:
        """
        Chunks documents that exceed token limit with recursive binary splitting.
        
        Args:
            documents: List of langchain Document objects
            max_tokens: Maximum number of tokens allowed per document
            overlap_percentage: Percentage of overlap between chunks
            
        Returns:
            New list of Document objects, with chunked documents and proper metadata
        """
        def process_document(doc, chunk_idx=0):
            """
            Recursively process a document, splitting if necessary.
            Returns a list of documents.
            """
            token_count = self.count_tokens(doc.page_content)
            
            # If document is within token limit, add chunk_index and return
            if token_count <= max_tokens:
                # Create new metadata with chunk_index
                new_metadata = doc.metadata.copy() if doc.metadata else {}
                new_metadata["chunk_index"] = chunk_idx
                new_metadata["token_count"] = token_count
                new_metadata["content_length"] = len(doc.page_content)
                
                return [Document(
                    page_content=doc.page_content,
                    metadata=new_metadata
                )]
            
            # If document exceeds token limit, split recursively
            first_half, second_half = self.split_text(doc.page_content, overlap_percentage)
            
            # Create documents from the halves
            first_doc = Document(
                page_content=first_half,
                metadata=doc.metadata.copy() if doc.metadata else {}
            )
            
            second_doc = Document(
                page_content=second_half,
                metadata=doc.metadata.copy() if doc.metadata else {}
            )
            
            # Process both halves recursively
            processed_first = process_document(first_doc, chunk_idx=f"{chunk_idx}.1")
            processed_second = process_document(second_doc, chunk_idx=f"{chunk_idx}.2")
            
            # Combine and return all resulting documents
            return processed_first + processed_second
        
        # Process all documents
        chunked_docs = []
        for i, doc in enumerate(documents):
            chunked_docs.extend(process_document(doc, chunk_idx=i))
            
        # Print chunking statistics
        original_count = len(documents)
        new_count = len(chunked_docs)
        chunked_count = sum(1 for doc in chunked_docs if "." in str(doc.metadata.get("chunk_index", "")))
        
        print(f"Original documents: {original_count}")
        print(f"After chunking: {new_count}")
        print(f"Documents that needed chunking: {chunked_count}")
        
        return chunked_docs
    
    def process_all_documents(self) -> List[Document]:
        """
        Process all documents from all sources and chunk them appropriately.
        
        Returns:
            List of processed and chunked documents
        """
        print("Starting document processing...")
        
        # Gather documents from all sources
        docs_pdf = self.parse_pdf_text_files()
        docs_raw = self.create_langchain_documents_raw()
        docs_results = self.create_langchain_documents_results()
        
        # Combine all documents
        unified_docs = docs_pdf + docs_raw + docs_results
        
        if not unified_docs:
            print("No documents found for processing.")
            return []
            
        print(f"Combined {len(unified_docs)} documents from all sources.")
        
        # Chunk documents based on token count
        chunked_docs = self.chunk_documents_by_tokens(unified_docs)
        
        print(f"Document processing complete. Total processed documents: {len(chunked_docs)}")
        return chunked_docs


# For standalone execution
def main():
    processor = DataProcessor()
    chunked_docs = processor.process_all_documents()
    print(f"Processed {len(chunked_docs)} documents total")
    return chunked_docs

if __name__ == "__main__":
    main()