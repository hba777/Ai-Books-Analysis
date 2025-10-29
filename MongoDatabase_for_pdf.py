import pdfplumber
import pymongo # PyMongo import kiya
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
load_dotenv()

# --- MongoDB Configuration ---
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME=os.getenv("MONGO_DB_PDF")
MONGO_COLLECTION_NAME=os.getenv("MONGO_COLLECTION_PDF")


# Initialize MongoDB client and collection
def init_mongo_db():
    client = pymongo.MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION_NAME]
    print(f"Connected to MongoDB: Database '{MONGO_DB_NAME}', Collection '{MONGO_COLLECTION_NAME}'")
    return client, collection

# Configuration
BOOK_TITLE = "History Book"
PDF_FILE = 'The Lost War.pdf'
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50

# Initialize LangChain text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""]
)

# Main logic to extract and store PDF chunks to MongoDB
def process_pdf_to_mongodb():
    mongo_client, mongo_collection = init_mongo_db()

    try:
        # Optional: Clear existing data from the collection before inserting new
        print(f"\n--- Clearing existing data in '{MONGO_COLLECTION_NAME}' collection ---")
        delete_result = mongo_collection.delete_many({})
        print(f"Deleted {delete_result.deleted_count} existing documents.")

        print(f"\n--- Processing PDF: {PDF_FILE} ---")
        documents_to_insert = []
        with pdfplumber.open(PDF_FILE) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    print(f"\n--- Page {page_num} ---")
                    chunks = text_splitter.split_text(page_text)
                    for chunk_num, chunk in enumerate(chunks, start=1):
                        print(f"Chunk {chunk_num}:\n{chunk}\n")

                        # Prepare the document for MongoDB
                        document = {
                            "book_title": BOOK_TITLE,
                            "page_number": page_num,
                            "chunk_number": chunk_num,
                            "chunk_text": chunk
                        }
                        documents_to_insert.append(document)

        if documents_to_insert:
            # Insert all documents at once for efficiency
            insert_result = mongo_collection.insert_many(documents_to_insert)
            print(f"\n✅ Successfully inserted {len(insert_result.inserted_ids)} chunks into MongoDB.")
        else:
            print("No text extracted or no chunks generated to insert.")

    except pymongo.errors.ConnectionFailure as e:
        print(f"❌ MongoDB connection error: {e}")
        print("Please ensure your MongoDB server is running on localhost:27017.")
    except FileNotFoundError:
        print(f"❌ Error: PDF file not found at '{PDF_FILE}'. Please check the path.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
    finally:
        if mongo_client:
            mongo_client.close()
            print("MongoDB connection closed.")

# Run the script
if __name__ == "__main__":
    process_pdf_to_mongodb()
