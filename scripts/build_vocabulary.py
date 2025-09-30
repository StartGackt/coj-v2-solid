
import sys
import os
import re

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pythainlp.tokenize import word_tokenize
from app.repositories.legal_ontology_repository import LegalOntologyRepository
from app.api.dependencies import SessionLocal
from sqlalchemy.exc import IntegrityError

# Define the path to the data file
DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "app", "dataset", "data1.txt")

def main():
    """
    Reads a large text file, tokenizes it into words using PyThaiNLP, 
    and saves the unique words into the relational database to build a vocabulary.
    """
    # 1. Read and Tokenize the entire file
    print(f"Reading data from {DATA_FILE_PATH}...")
    try:
        with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_FILE_PATH}")
        return

    print("Tokenizing text into words with PyThaiNLP...")
    words = word_tokenize(text)
    
    # Clean up and get unique words
    # This regex keeps Thai, English, and numbers. It removes punctuation and spaces.
    clean_words = {word for word in words if re.match(r'^[฀-๿\w\d]+$', word)}
    
    print(f"Found {len(clean_words)} unique words to process.")

    # 2. Save to Database
    print("\nConnecting to PostgreSQL and saving new words...")
    saved_count = 0
    skipped_count = 0
    db_session = None
    try:
        db_session = SessionLocal()
        repo = LegalOntologyRepository(db=db_session)
        
        for i, word in enumerate(clean_words):
            try:
                repo.create({"name": word, "description": "legal_term"})
                saved_count += 1
                # Print progress occasionally
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(clean_words)} words...")
            except IntegrityError:
                db_session.rollback()
                skipped_count += 1

        print("\n--- Vocabulary Build Summary ---")
        print(f"Successfully saved {saved_count} new words to the vocabulary.")
        print(f"Skipped {skipped_count} words that already existed.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        if db_session:
            db_session.close()

if __name__ == "__main__":
    main()
