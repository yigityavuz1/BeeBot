from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# Context analysis prompt to check if the context is sufficient
CONTEXT_ANALYSIS_PROMPT = ChatPromptTemplate.from_template(
    """Sen, İstanbul Teknik Üniversitesi (İTÜ) öğrencilerinin İTÜ hakkındaki sorularını yanıtlayan bir eğitim asistanısın.\
    Aşağıda verilen bağlamın, soruya eksiksiz ve doğru cevap verebilmek için yeterli olup olmadığını değerlendir.\

    Soru: {query}

    Bağlam:
    {context}

    Verilen bağlam, soruya tam ve doğru bir cevap sunmak için yeterli midir?
    Cevabı yalnızca {{"sufficient": true}} veya {{"sufficient": false}} şeklinde JSON formatında ver.
    """
)

# System prompt for the RAG assistant
SYSTEM_PROMPT = """Sen Türkçe konuşan, İstanbul Teknik Üniversitesi (İTÜ) öğrencilerinin İTÜ hakkındaki sorularını sana verilen bilgilere göre cevaplayan bir yardım asistanısın. 
Kullanıcı soruyu sorduğunda, yalnızca sana verilen bağlamdaki bilgiyi kullanarak Türkçe yanıt ver. 
Eğer bağlamda yeterli bilgi bulunamazsa, bunu açıkça belirt."""

# Context prompt for formatting the context
CONTEXT_PROMPT = ChatPromptTemplate.from_template("""
### BAĞLAM - BELGE 1 (alpha=0.75)
{strict_content}

### BAĞLAM - BELGE 2 (alpha=0.5)
{relaxed_content}

{web_content_section}

### SORU
{query}

### TALİMAT
- Sadece yukarıdaki bağlamdaki bilgiye dayanarak cevap ver.
- Bağlamda olmayan bilgiyi ekleme.
- Yanıtı Türkçe olarak ver.
- Birden çok bağlam türü varsa, tüm bağlamlardan ilgili bilgileri kullanarak kapsamlı bir yanıt oluştur.
- Yanıtlar için kaynak türünü ve güven skorunu da belirt.
""")

# Helper template for JSON formatting
COMPLETION_HELPER = """
Lütfen cevabını aşağıdaki JSON formatında ver:
{{
  "answer": "Sorunun detaylı cevabı",
  "source": "{source_type}",
  "confidence": 0.8 (Cevabın doğruluğuna dair 0-1 arası güven skoru)
}}
"""

def get_response_messages(system_prompt, context_prompt, source_type, strict_content, relaxed_content, web_content, query):
    """Create the messages for the LLM response generation."""
    # Prepare web content section
    web_content_section = f"### WEB ARAMA SONUÇLARI\n{web_content}" if web_content else ""
    
    # Create the messages
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=context_prompt.format(
            strict_content=strict_content,
            relaxed_content=relaxed_content,
            web_content_section=web_content_section,
            query=query
        )),
        HumanMessage(content=COMPLETION_HELPER.format(source_type=source_type))
    ]
    
    return messages