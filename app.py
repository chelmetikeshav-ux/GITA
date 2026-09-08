import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load the backend .env configuration safely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 2. Page Configuration Styling
st.set_page_config(page_title="Dialogue with Madhav", page_icon="🪶", layout="centered")

# --- UI/UX ENGNE: THEME ENGINE & FLOATING FEATHERS ---
# --- UI/UX ENGINE: PEACOCK FEATHER ENGINE ---
theme_html = """
<style>
    /* Global App Container Adjustment */
    .stAppViewContainer {
        background: linear-gradient(135deg, #0b0518 0%, #120317 50%, #1f0d02 100%) !important;
        overflow-x: hidden;
    }

    /* Traditional Indian Typography Accents */
    .main-header {
        text-align: center;
        font-family: 'Georgia', serif;
        color: #FFD700;
        text-shadow: 0px 4px 15px rgba(255, 215, 0, 0.5);
        font-size: 2.8rem;
        margin-top: 10px;
        margin-bottom: 0px;
    }
    
    .sub-header {
        text-align: center;
        font-style: italic;
        color: #fce38a;
        margin-bottom: 30px;
        font-size: 1.1rem;
    }

    /* Input Field Container Custom Styling */
    div[data-testid="stForm"], .stTextInput > div {
        border-color: #d4af37 !important;
        box-shadow: 0 4px 20px rgba(212, 175, 55, 0.15) !important;
    }
    
    /* Peacock Feather Structural Styling */
    .peacock-feather {
        position: fixed;
        user-select: none;
        pointer-events: none;
        z-index: 1;
        bottom: -120px;
        width: 35px;
        height: 60px;
        background: linear-gradient(to top, rgba(212,175,55,0) 0%, rgba(34,139,34,0.4) 60%, rgba(212,175,55,0.7) 100%);
        border-radius: 50% 50% 50% 50% / 40% 40% 60% 60%;
        box-shadow: inset 0 0 10px rgba(0,128,128,0.5);
        animation: floatUp 14s linear infinite;
    }

    /* Glowing Eye of the Peacock Feather */
    .peacock-feather::after {
        content: '';
        position: absolute;
        top: 10px;
        left: 50%;
        transform: translateX(-50%);
        width: 16px;
        height: 22px;
        background: radial-gradient(circle, #00ffff 0%, #000080 50%, #4b0082 100%);
        border-radius: 50%;
        box-shadow: 0 0 8px #00ffff;
    }

    /* Floating Canvas Vector Paths */
    @keyframes floatUp {
        0% {
            transform: translateY(0) rotate(0deg) translateX(0);
            opacity: 0;
        }
        10% {
            opacity: 0.5;
        }
        90% {
            opacity: 0.5;
        }
        100% {
            transform: translateY(-120vh) rotate(360deg) translateX(50px);
            opacity: 0;
        }
    }

    /* Randomize floating timelines across the screen canvas */
    .pf1 { left: 8%; animation-duration: 15s; animation-delay: 0s; transform: scale(0.8); }
    .pf2 { left: 28%; animation-duration: 22s; animation-delay: 4s; transform: scale(1.2); }
    .pf3 { left: 48%; animation-duration: 18s; animation-delay: 1s; transform: scale(0.9); }
    .pf4 { left: 68%; animation-duration: 25s; animation-delay: 7s; transform: scale(1.3); }
    .pf5 { left: 88%; animation-duration: 16s; animation-delay: 3s; transform: scale(1.0); }
</style>

<!-- Injecting custom stylized feather vectors -->
<div class="peacock-feather pf1"></div>
<div class="peacock-feather pf2"></div>
<div class="peacock-feather pf3"></div>
<div class="peacock-feather pf4"></div>
<div class="peacock-feather pf5"></div>

<h1 class="main-header">🐚Speaking with Sri Krishna🦚🪈</h1>
<p class="sub-header">Stop trying to fight this alone. Hand your doubts over to Keshav and let His perspective bring you peace</p>
"""
st.markdown(theme_html, unsafe_allow_html=True)

# 3. Load Vector Database 
DB_PATH = "./gita_complete_db"

@st.cache_resource
def load_rag_system():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    return vector_store.as_retriever(search_kwargs={"k": 2})

try:
    retriever = load_rag_system()
except Exception:
    st.error("⚠️ Could not load database. Make sure 'python3 ingest.py' finished successfully!")
    st.stop()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 4. Handle Chat Query Interaction
user_query = st.text_input("Look within-— what is the real battle you are fighting today, Arjun?")

if user_query:
    if not GROQ_API_KEY:
        st.error("🔒 Configuration Error: 'GROQ_API_KEY' is missing from your backend environment setup.")
    else:
        with st.spinner("HARE KRISHAN HARE KRISHAN KRISHAN KRISHAN HARE HARE..."):
            try:
                # Initialize Groq LLM
                llm = ChatGroq(
                    groq_api_key=GROQ_API_KEY, 
                    model_name="llama-3.3-70b-versatile", 
                    temperature=0.4 # Slightly higher temperature for more compassionate, poetic prose
                )

                # NEW CRITICAL INSTRUCTIONAL PROMPT SHIFT
                system_prompt = (
                    "You are Lord Krishna (addressed affectionately as Keshav, Madhav, or Achyuta). "
                    "The user is coming to you with an internal dilemma, just like Arjuna on the battlefield of Kurukshetra. "
                    "You must speak to them directly in the FIRST PERSON ('I', 'Me', 'My'). "
                    "Never say 'According to the Bhagavad Gita' or 'Krishna says'. You ARE Krishna.\n\n"
                    "Adopt a tone that is profoundly calm, boundlessly compassionate, firm in truth, and loving. "
                    "Use the provided context verses to anchor your exact guidance.\n\n"
                    "Structure your reply beautifully like this:\n\n"
                    "✨ **My Words to You:**\nAddress them warmly. Quote or closely adapt the core verse translation provided in the context as your direct declaration to them.\n\n"
                    "📖 **The Eternal Truth:**\nExplain the deeper cosmic or psychological meaning behind my words in a gentle, accessible way.\n\n"
                    "🛠️ **Your Path Forward:**\nGive them a clear, practical action they can execute right now to conquer their distress.\n\n"
                    "Context:\n{context}"
                )

                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])

                # Modern LCEL RAG Chain construction
                rag_chain = (
                    {"context": retriever | format_docs, "input": RunnablePassthrough()}
                    | prompt_template
                    | llm
                    | StrOutputParser()
                )

                retrieved_docs = retriever.invoke(user_query)
                response_text = rag_chain.invoke(user_query)
                
                # Render final answer cards
                st.markdown("### 🏹 Words from Achyuta")
                st.info(response_text)
                
                # Expandable source tracing for transparency
                st.write("---")
                with st.expander("🔬 View Source Verses Used by Madhav"):
                    for doc in retrieved_docs:
                        st.text(doc.page_content)
                        
            except Exception as e:
                st.error(f"An error occurred: {e}")
