# config.py
import os
from dotenv import load_dotenv
load_dotenv()

# --- MongoDB Configuration ---
MONGO_URI = os.getenv("MONGO_URI")
AGENTS_DB_NAME = os.getenv("MONGO_DB_Agent")
AGENTS_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_Agent")

KB_DB_NAME = os.getenv("MONGO_DB_KB")
KB_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_KB")

# --- Modified: Align PDF chunk storage with Pipeline 1 ---
# Pipeline 1 stores chunks in the database named "document_classification"
# and in the 'chunks' collection.
PDF_DB_NAME = "document_classification" # Explicitly set to Pipeline 1's database name
PDF_COLLECTION_NAME = "chunks" # This is the collection name Pipeline 1 uses for chunks

# --- ChromaDB Persistent Directory ---
CHROMA_DB_DIRECTORY = "chrome_dB"

# Replace with your actual API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY1")