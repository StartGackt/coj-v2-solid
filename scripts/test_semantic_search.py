
import sys
import os
import re

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
import chromadb

# Define the path to the data file
DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "app", "dataset", "data1.txt")

def main():
    """
    Demonstrates semantic search by creating vector embeddings for legal articles
    and searching for them based on a query's meaning.
    """
    # 1. Load a pre-trained model for creating sentence embeddings
    print("Loading sentence-transformer model... (This may take a moment)")
    # Using a multilingual model is a good choice for Thai
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("Model loaded.")

    # 2. Read and parse the legal text into articles (มาตรา)
    print(f"Reading and parsing legal articles from {DATA_FILE_PATH}...")
    try:
        with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
            full_text = f.read()
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_FILE_PATH}")
        return

    # Split the text into articles based on "มาตรา"
    # We use a regex to handle variations in spacing and numbering
    articles = re.split(r'\n\s*มาตรา[\s๐-๙\d/]+', full_text)
    # Get the article numbers/names as well
    article_titles = [title.strip() for title in re.findall(r'มาตรา[\s๐-๙\d/]+', full_text)]
    
    # Filter out any empty strings that result from splitting
    # and associate titles with content
    documents = {
        title: content.strip()
        for title, content in zip(article_titles, articles[1:]) # articles[0] is usually empty
        if content.strip()
    }
    print(f"Parsed {len(documents)} articles.")

    # 3. Setup ChromaDB (Vector Database)
    # Using an in-memory instance for this demonstration
    client = chromadb.Client()
    # Create a collection to store our vectors. Use cosine distance for sentence-transformers.
    collection = client.get_or_create_collection(
        name="legal_articles",
        metadata={"hnsw:space": "cosine"} 
    )
    print("Vector database collection is ready.")

    # 4. Generate and store embeddings for each article
    print("Generating and storing vector embeddings for each article...")
    ids = list(documents.keys())
    contents = list(documents.values())
    
    # Generate embeddings in batches (more efficient for large numbers of docs)
    embeddings = model.encode(contents, show_progress_bar=True)
    
    # Add to ChromaDB
    collection.add(
        embeddings=embeddings,
        documents=contents,
        ids=ids
    )
    print("All articles have been embedded and stored.")

    # 5. Perform Semantic Search
    print("\n--- Semantic Search Test ---")
    
    # The query doesn't need to use exact keywords from the text
    query_text = "การเลิกจ้างเพราะลูกจ้างผู้หญิงตั้งท้อง"
    
    print(f"Searching for: '{query_text}'")
    
    # Generate the embedding for the query
    query_embedding = model.encode([query_text])
    
    # Query the collection to find the most similar articles
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3 # Ask for the top 3 most similar results
    )
    
    print("\nTop 3 most relevant articles found:")
    for i, (doc_id, distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
        print(f"{i+1}. {doc_id} (Similarity Score: {1 - distance:.4f})")
        # print(f"   Content: {results['documents'][0][i][:100]}...") # Optionally print content snippet

if __name__ == "__main__":
    main()
