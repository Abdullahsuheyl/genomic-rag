## Project Overview

This project is a RAG system that automatically fetches scientific articles related to the RARS1 gene from PubMed, indexes them into ChromaDB, and performs question-answering via the Gemini LLM.
To run this project, first install the necessary dependencies using the requirements file, and then execute the main script: 
```bash
    pip install -r requirements.txt
    python main.py
```

### Workflow

The process begins by retrieving up-to-date biomedical articles via the PubMed API. The ingest.py script handles the integration of this raw data into the system.
Since large research papers are too long to be processed directly by a language model, they are divided into smaller, meaningful chunks using the RecursiveCharacterTextSplitter. This method attempts to preserve paragraph and sentence integrity during splitting to minimize context loss.
These text chunks are then converted into numerical vectors using the S-BioBERT model, which is specialized in biomedical terminology. Unlike standard BERT models, it is highly proficient in medical language. These vectors are stored in a ChromaDB vector database, allowing the system to rapidly retrieve article segments that are semantically closest to a user's query.
When a user asks a question, the query is also vectorized, and a similarity search is performed within ChromaDB. The relevant article sections are retrieved and combined with the user's question before being sent to the Gemini LLM. Gemini uses this specific scientific data to generate a structured response.

### Design Decisions
#### 1. PubMed API Rate Limit Management

The NCBI's free tier is limited to 3 requests per second without authentication. To prevent potential blocking and define API access, an email address is required for identification by NCBI. The system first retrieves a list of IDs and then fetches all articles in a single XML response. This approach avoids making separate requests for every individual article, keeping the request count to a minimum. Once retrieved, data is written to ChromaDB with a 0.1-second delay between batches of 50 to ensure stability.
#### 2. Embedding Model

The pritamdeka/S-BioBert-snli-multinli-stsb model was chosen for this project. Based on the BioBERT architecture, this model has been pre-trained on PubMed articles and PMC full-texts. Consequently, it possesses a much stronger representation capacity for medical terminology, gene names, and clinical concepts compared to general-purpose models. It has been fine-tuned using the Sentence-Transformers framework on SNLI, MultiNLI, and STS-B datasets, optimizing it for dual-sentence comparison and semantic similarity tasks. This ensures more accurate query-passage matching, which is critical for RAG systems.
#### 3. Phenotype and Variant Differentiation

A prompt-engineering-based approach is utilized to ensure the LLM presents information regarding RARS1 in a structured format. Gemini 2.5 Flash is the preferred model for this task. Every response is required to have two mandatory sections:
•	Phenotypes / Clinical Features
•	Variants / Mutations
Furthermore, a citation requirement in the format of [PMID: XXXXX] is enforced for every scientific claim. This rule prevents the LLM from conflating phenotype and variant information and ensures that every statement is grounded in a concrete research paper. If the question is unrelated to RARS1 or if the answer is not present in the provided literature, the LLM is instructed to respond only with: "I do not know based on available literature."
