import time
import logging
from Bio import Entrez
from langchain_text_splitters import RecursiveCharacterTextSplitter


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Entrez.email = "asuheylozmen0@gmail.com"

def fetch_pubmed_abstracts(query="RARS1", max_results=40):
    """Programmatically query PubMed and fetch latest abstracts"""
    try:
        # Handling rate limits
        search_handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
        search_results = Entrez.read(search_handle)
        search_handle.close()
        
        id_list = search_results.get("IdList", [])
        if not id_list:
            logger.warning("No articles found for the given query.")
            return []

        fetch_handle = Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="abstract", retmode="xml")
        articles = Entrez.read(fetch_handle)
        fetch_handle.close()
        
        return articles.get('PubmedArticle', [])
    except Exception as e:
        logger.error(f"API Error: {e}")
        return []

def get_processed_chunks(articles):
    """Smart chunking to keep medical terms."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        separators=["\n\n", "\n", "(?<=[.?!]) +", " ", ""],
        is_separator_regex=True
    )
    
    final_chunks = []
    for article in articles:
        pmid = str(article['MedlineCitation']['PMID'])
        abstract_parts = article['MedlineCitation']['Article'].get('Abstract', {}).get('AbstractText', [])
        content = " ".join(abstract_parts)
        
        if not content: continue

        chunks = splitter.split_text(content)
        for chunk in chunks:
            final_chunks.append({
                "text": chunk,
                "metadata": {"source": pmid} 
            })
    return final_chunks