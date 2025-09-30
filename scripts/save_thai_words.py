
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pythainlp.tokenize import word_tokenize
from app.repositories.legal_ontology_repository import LegalOntologyRepository
from app.api.dependencies import SessionLocal
from sqlalchemy.exc import IntegrityError

def main():
    """
    Tokenizes a sentence into whole words using PyThaiNLP and saves each word 
    into the relational database.
    """
    # 1. Tokenize into words
    text_to_tokenize = "นายจ้างต้องคืนหลักประกันให้ลูกจ้างภายใน 7 วัน"
    
    words = word_tokenize(text_to_tokenize)
    # Filter out empty strings or spaces that might result from tokenization
    words = [word for word in words if word.strip()]

    print(f"Found {len(words)} words to save.")
    print(words)

    # 2. Save to Database
    print("\nConnecting to PostgreSQL and saving words...")
    saved_count = 0
    skipped_count = 0
    db_session = None
    try:
        db_session = SessionLocal()
        repo = LegalOntologyRepository(db=db_session)
        
        for word in words:
            try:
                # Use the word as the 'name' and 'word' as the description
                repo.create({"name": word, "description": "word"})
                print(f"  - Saved: {word}")
                saved_count += 1
            except IntegrityError:
                # This happens if the word already exists (due to unique constraint)
                db_session.rollback() # Rollback the failed transaction
                print(f"  - Skipped (already exists): {word}")
                skipped_count += 1

        print("\n--- Summary ---")
        print(f"Successfully saved {saved_count} new words.")
        print(f"Skipped {skipped_count} duplicate words.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        if db_session:
            db_session.close()

if __name__ == "__main__":
    main()
