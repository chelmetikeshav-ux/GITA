import os
import json
import glob
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

INPUT_FOLDER = "./gita_chapters"
DB_PATH = "./gita_complete_db"

def build_vector_db():
    json_files = glob.glob(os.path.join(INPUT_FOLDER, "*.json"))
    
    if not json_files:
        print(f"Error: No JSON files found in '{INPUT_FOLDER}'. Please add your files.")
        return

    print(f"Found {len(json_files)} chapter files. Processing...")
    langchain_documents = []

    for file_path in json_files:
        filename = os.path.basename(file_path)
        
        with open(file_path, "r", encoding="utf-8") as f:
            chapter_data = json.load(f)
            verses_list = chapter_data if isinstance(chapter_data, list) else chapter_data.get("verses", [])
            
            for entry in verses_list:
                # Get the verse/chapter info safely for debugging
                v_num = entry.get("verse", "Unknown")
                c_num = entry.get("chapter", "Unknown")
                
                # Using .get() prevents KeyError crashes if a field is missing
                translation = entry.get("translation", "No translation provided.")
                explanation = entry.get("simple_explanation", "No explanation provided.")
                life_app = entry.get("life_application")
                keywords_list = entry.get("keywords", [])
                theme = entry.get("theme", "General")
                
                # Warning diagnostic if life_application is missing
                if life_app is None:
                    print(f"⚠️ Warning: Missing 'life_application' key in {filename} (Verse {v_num})")
                    life_app = "No life application provided."
                
                # Combine fields safely
                page_content = (
                    f"Chapter {c_num}, Verse {v_num}\n"
                    f"Translation: {translation}\n"
                    f"Explanation: {explanation}\n"
                    f"Life Application: {life_app}\n"
                    f"Keywords: {', '.join(keywords_list) if isinstance(keywords_list, list) else ''}"
                )
                
                metadata = {
                    "id": entry.get("id", f"{c_num}_{v_num}"),
                    "chapter": int(c_num) if str(c_num).isdigit() else 0,
                    "verse": int(v_num) if str(v_num).isdigit() else 0,
                    "theme": theme
                }
                
                langchain_documents.append(Document(page_content=page_content, metadata=metadata))

    print(f"\nAll files parsed successfully! Loaded {len(langchain_documents)} verses. Building database...")
    print("Generating embeddings... This might take a few moments.")
    
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    Chroma.from_documents(
        documents=langchain_documents,
        embedding=embedding_model,
        persist_directory=DB_PATH
    )
    print(f"🚀 Success! Vector database permanently saved to '{DB_PATH}'")

if __name__ == "__main__":
    build_vector_db()
