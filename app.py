import os
import json
import asyncio
import streamlit as st
st.set_page_config(
    page_title="BeeBot: İTÜ Bilgi Asistanı",
    page_icon="🐝",
    layout="wide"
)
from dotenv import load_dotenv

from vectordb import VectorDatabase
from rag_graph import EnhancedRAGSystem, extended_process_query
from tools import format_context, format_metadata, run_async

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize session state for resources
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "loop" not in st.session_state:
    # Create a new event loop
    st.session_state.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(st.session_state.loop)


async def init_vector_db():
    """Initialize the vector database."""
    if st.session_state.vector_db is None:
        with st.spinner("Vector DB yükleniyor... Bu ilk kez ise birkaç dakika sürebilir."):
            # Initialize the vector database
            vector_db = VectorDatabase(openai_api_key=openai_api_key)
            await vector_db.connect()
            await vector_db.create_collection()
            
            # Store the vector database in session state
            st.session_state.vector_db = vector_db
    
    return st.session_state.vector_db


async def init_rag_system():
    """Initialize the RAG system."""
    if st.session_state.rag_system is None:
        # Ensure vector database is initialized
        vector_db = await init_vector_db()
        
        # Initialize the RAG system
        rag_system = EnhancedRAGSystem(openai_api_key=openai_api_key)
        
        # Create a retriever from the vector database
        retriever = await vector_db.create_retriever()
        
        # Create the graph with the retriever
        rag_system.create_graph(retriever)
        
        # Store the RAG system in session state
        st.session_state.rag_system = rag_system
    
    return st.session_state.rag_system


def main_ui():
    st.title("🐝BeeBot: İTÜ Bilgi Asistanı")
    
    # Önerilen sorular
    önerilen_sorular = [
        "Mimarlığa yatay geçiş için not ortalamam kaç olmalı?",
        "2024 akademik takviminde önemli tarihler ve sınav dönemleri nelerdir?",
        "Lisans ve lisansüstü başvuru süreçlerinde hangi adımlar takip ediliyor?",
        "Yabancı dil sınavı başvuru adımları ve gereken belgeler nelerdir?",
        "OBS'ye giriş, şifre sıfırlama ve diğer kullanıcı işlemleri nasıl gerçekleştiriliyor?",
        "İTÜ'nün ders başarı ölçme, değerlendirme ve not dönüştürme esasları nelerdir?",
        "Mezuniyet sonrası diploma alım prosedürü hangi adımlardan oluşuyor?",
        "Diploma belgesi almak için gereken belgeler ve onay süreçleri nedir?",
        "Ders denkliklerinin ve kredi transferlerinin onaylanması için hangi belgeler gereklidir?"
    ]
    
    st.sidebar.header("Önerilen Sorular")
    
    # Initialize the RAG system
    with st.spinner("Vector DB yükleniyor..."):
        try:
            rag_system = run_async(st.session_state.loop, init_rag_system())
        except Exception as e:
            st.error(f"Sistem hazırlanırken hata oluştu: {str(e)}")
            return
    
    # Handle sidebar buttons for recommended questions
    for soru in önerilen_sorular:
        if st.sidebar.button(soru):
            process_and_display_response(soru, rag_system)
    
    # User query section
    st.header("Yeni Soru Sor")
    user_query = st.text_input("Soru:")
    if st.button("Cevapla") and user_query:
        process_and_display_response(user_query, rag_system)


def process_and_display_response(query, rag_system):
    """Process the query and display the response."""
    with st.spinner("Cevap hazırlanıyor..."):
        try:
            # Process the query
            response, context_info = run_async(
                st.session_state.loop, 
                extended_process_query(query, rag_system)
            )
            
            # Display the results
            st.write("### Soru:")
            st.write(query)
            st.write("### Cevap:")
            st.write(response["answer"])
            st.write("### Kullanılan Bağlam:")
            st.markdown(format_context(context_info))
            st.write("### Metadata:")
            st.code(format_metadata(response))
        except Exception as e:
            st.error(f"Soru işlenirken bir hata oluştu: {str(e)}")


if __name__ == "__main__":
    main_ui()