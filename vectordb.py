import os
import asyncio
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
import streamlit as st
from tools import logger

class VectorDatabase:
    """Class for managing Weaviate vector database operations."""
    
    def __init__(self, openai_api_key):
        """Initialize the vector database with API token."""
        self.openai_api_key = openai_api_key
        self.client = None
        self.collection_name = "beebot_index"
        self.embeddings = None
        self._connection_lock = asyncio.Lock()
        self._is_connected = False
        logger.info("VectorDatabase instance initialized")
    
    async def connect(self):
        """Connect to Weaviate database."""
        async with self._connection_lock:
            if self._is_connected and self.client:
                logger.info("Already connected to Weaviate")
                return self.client
            
            # Get connection details from environment variables with defaults
            weaviate_host = os.getenv("WEAVIATE_HOST", "localhost")
            weaviate_port = os.getenv("WEAVIATE_PORT", "8080")
            weaviate_port_grpc = os.getenv("WEAVIATE_PORT_GRPC", "50051")
            
            logger.info(f"Connecting to Weaviate at {weaviate_host}:{weaviate_port}")
            
            # Use custom connection with host and port from environment
            try:
                self.client = weaviate.use_async_with_custom(
                    http_host=weaviate_host,
                    http_port=int(weaviate_port),
                    http_secure=False,
                    grpc_host=weaviate_host,
                    grpc_port=weaviate_port_grpc,
                    grpc_secure=False
                )
                await self.client.connect()
                await self.client.is_ready()
                
                # Initialize embeddings
                logger.info("Initializing OpenAI embeddings")
                self.embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-large",
                    api_key=self.openai_api_key
                )
                
                self._is_connected = True
                logger.info("Successfully connected to Weaviate")
                return self.client
            
            except Exception as e:
                error_msg = f"Failed to connect to Weaviate: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)
                raise
    
    async def ensure_connected(self):
        """Ensure the client is connected before operations."""
        if not self._is_connected or not self.client:
            logger.info("Connection check failed, reconnecting to Weaviate")
            await self.connect()
        else:
            logger.debug("Connection check passed")
    
    async def collection_exists(self):
        """Check if the collection exists."""
        await self.ensure_connected()
        try:
            logger.info(f"Checking if collection '{self.collection_name}' exists")
            collections = await self.client.collections.list_all()
            exists = bool(collections.get(self.collection_name.capitalize()))
            logger.info(f"Collection '{self.collection_name}' exists: {exists}")
            return exists
        except Exception as e:
            error_msg = f"Error checking collection existence: {str(e)}"
            logger.error(error_msg)
            st.warning(error_msg)
            return False
    
    async def create_collection(self):
        """Create Weaviate collection if it doesn't exist."""
        await self.ensure_connected()
        
        # Check if collection exists first
        collection_exists = await self.collection_exists()
        if collection_exists:
            logger.info(f"Collection '{self.collection_name}' already exists. Skipping creation.")
            return True
        
        logger.info(f"Collection '{self.collection_name}' doesn't exist. Creating new collection...")
        
        # Define vectorizer configuration
        vectorizer_config = [
            Configure.NamedVectors.text2vec_openai(
                name="page_content_vectorizer",
                source_properties=["page_content"],
                model="text-embedding-3-large",
            )
        ]

        try:
            logger.info("Creating Weaviate collection with OpenAI vectorizer")
            await self.client.collections.create(
                name=self.collection_name,
                vectorizer_config=vectorizer_config,
                properties=[
                    # The one text field to be vectorized
                    Property(
                        name="page_content",
                        data_type=DataType.TEXT,
                        description="Transcript chunk text"
                    ),
                    Property(
                        name="page_content_vector",
                        data_type=DataType.NUMBER_ARRAY,
                        description="Embedding for the transcript chunk text"
                    ),
                    # The remaining fields as normal (non-vector) properties
                    Property(
                        name="source",
                        data_type=DataType.TEXT,
                        description="Source identifier"
                    ),
                    Property(
                        name="chunk_length",
                        data_type=DataType.NUMBER,
                        description="Number of characters in this chunk"
                    ),
                    Property(
                        name="chunk_index",
                        data_type=DataType.TEXT,
                        description="Index of this chunk"
                    ),
                    Property(
                        name="token_count",
                        data_type=DataType.NUMBER,
                        description="Number of tokens in this chunk"
                    ),
                ],
            )
            logger.info(f"Collection '{self.collection_name}' created successfully.")
            
            # Since the collection was just created, populate it with data
            logger.info("Populating newly created collection with data")
            await self.populate_collection()
            
        except Exception as e:
            if "ResourceNameAlreadyInUse" in str(e):
                logger.warning(f"Collection {self.collection_name} already exists. Skipping creation.")
            else:
                error_msg = f"Error creating collection: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)
                raise e
        
        return True
    
    async def populate_collection(self):
        """Populate the collection with documents from data sources."""
        logger.info("Starting to populate collection with documents...")
        
        # Import here to avoid circular imports
        from process_data import DataProcessor
        
        # Process data using DataProcessor
        processor = DataProcessor()
        documents = processor.process_all_documents()
        
        if not documents:
            logger.warning("No documents found to populate the collection.")
            return
        
        # Get collection reference
        collection = self.client.collections.get(self.collection_name)
        
        # Upload documents in batches
        batch_size = 50
        total_documents = len(documents)
        
        logger.info(f"Uploading {total_documents} documents to Weaviate in batches of {batch_size}...")
        
        for i in range(0, total_documents, batch_size):
            batch = documents[i:i+batch_size]
            
            # Create batch with embeddings
            batch_objects = []
            for doc in batch:
                # Generate embedding for the document
                logger.debug(f"Generating embedding for document: {doc.metadata.get('source', 'unknown')}")
                embedding = await self.embeddings.aembed_query(doc.page_content)
                
                # Prepare object
                obj = {
                    "page_content": doc.page_content,
                    "page_content_vector": embedding,
                    "source": doc.metadata.get("source", ""),
                    "chunk_length": doc.metadata.get("content_length", len(doc.page_content)),
                    "chunk_index": str(doc.metadata.get("chunk_index", "")),
                    "token_count": doc.metadata.get("token_count", 0)
                }
                batch_objects.append(obj)
            
            # Upload batch
            try:
                logger.info(f"Uploading batch {i//batch_size + 1}/{(total_documents-1)//batch_size + 1}")
                with collection.batch as batch_writer:
                    for obj in batch_objects:
                        batch_writer.add_object(properties=obj)
                
                logger.info(f"Successfully uploaded batch {i//batch_size + 1}/{(total_documents-1)//batch_size + 1}")
            except Exception as e:
                logger.error(f"Error uploading batch: {e}")
        
        logger.info(f"Collection population complete. Added {total_documents} documents.")
    
    async def create_retriever(self):
        """Create a retriever function for the Weaviate collection."""
        await self.ensure_connected()
        logger.info("Creating document retriever function")
        collection = self.client.collections.get(self.collection_name)
        
        async def retrieve(query: str, k: int = 3, alpha: float = 0.5):
            """Retrieve relevant documents from Weaviate."""
            await self.ensure_connected()
            
            logger.info(f"Retrieving documents for query: '{query}' with k={k}, alpha={alpha}")
            query_embedding = await self.embeddings.aembed_query(query)
            logger.debug("Generated query embedding")
            
            results = await collection.query.hybrid(
                query=query, vector=query_embedding, limit=k, alpha=alpha,
            )
            
            logger.info(f"Retrieved {len(results.objects)} objects from Weaviate")
            
            documents = []
            seen_content = set()  # To deduplicate results
            
            for obj in results.objects:
                content = obj.properties.get('page_content', '')
                if content and content not in seen_content:
                    seen_content.add(content)
                    metadata = {
                        "source": obj.properties.get('source', ''),
                        "chunk_index": obj.properties.get('chunk_index', ''),
                    }
                    documents.append(Document(page_content=content, metadata=metadata))
            
            logger.info(f"Converted {len(documents)} unique documents from results")
            return documents
        
        logger.info("Document retriever function created successfully")
        return retrieve