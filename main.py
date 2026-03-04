import json
import time
import chromadb
import google.generativeai as genai
from chromadb.utils import embedding_functions
from ingest import fetch_pubmed_abstracts, get_processed_chunks
import warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
genai.configure(api_key=GEMINI_API_KEY)


MODEL_NAME = "gemini-2.5-flash"
REQUEST_DELAY = 5  


class GenomicRAG:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="pritamdeka/S-BioBert-snli-multinli-stsb"
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="rars1_research",
            embedding_function=self.embedding_fn
        )
        self.llm = genai.GenerativeModel(MODEL_NAME)

    def setup(self, force_reingest=False):
        existing_count = self.collection.count()

        if existing_count > 0 and not force_reingest:
            print(f"{existing_count} chunks already exist in ChromaDB. Skipping ingest.")
            print("To reload, use setup(force_reingest=True)\n")
            return

        if force_reingest:
            print("Deleting existing collection and recreating...")
            self.chroma_client.delete_collection("rars1_research")
            self.collection = self.chroma_client.get_or_create_collection(
                name="rars1_research",
                embedding_function=self.embedding_fn
            )

        print("Fetching RARS1 abstracts from PubMed.")
        articles = fetch_pubmed_abstracts()

        if not articles:
            print("No articles found. Please check your internet connection and API access.")
            return

        print(f"{len(articles)} articles found. Chunking in progress...")
        chunks = get_processed_chunks(articles)
        print(f"{len(chunks)} chunks created. Loading into ChromaDB...")

        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.collection.add(
                ids=[f"id_{i + j}" for j in range(len(batch))],
                documents=[c["text"] for c in batch],
                metadatas=[c["metadata"] for c in batch]
            )
            time.sleep(0.1)

        print(f"{len(chunks)} chunks successfully added to ChromaDB.\n")

    def _call_llm_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Calls the LLM with automatic retry on rate limit errors."""
        for attempt in range(max_retries):
            try:
                response = self.llm.generate_content(prompt)
                return response.text
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:               
                    wait_time = REQUEST_DELAY * (attempt + 1) * 10
                    try:
                        if "retry_delay" in error_str and "seconds:" in error_str:
                            seconds_part = error_str.split("seconds:")[1].strip()
                            wait_time = int(seconds_part.split()[0]) + 5
                    except Exception:
                        pass
                    print(f"  Quota exceeded. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    raise e
        return " Max retries exceeded due to rate limiting."

    def ask(self, query: str, n_results: int = 5) -> str:
        if self.collection.count() == 0:
            return "Database is empty. Please run setup() first."

        results = self.collection.query(query_texts=[query], n_results=n_results)

        context = ""
        for i, doc in enumerate(results["documents"][0]):
            pmid = results["metadatas"][0][i]["source"]
            context += f"[PMID: {pmid}]: {doc}\n\n"

        prompt = f"""You are a specialized genomics assistant. Answer ONLY based on the provided sources.

STRICT RULES:
1. Every scientific claim MUST include [PMID: XXXXX] citation.
2. If the question is NOT about RARS1, or the answer is not in the sources, respond with
   ONLY this exact sentence and nothing else: "I do not know based on available literature."
   Do NOT add any additional information, context, or related facts after this sentence.
3. Structure your response with two clear sections:
   - **Phenotypes / Clinical Features**: Observable symptoms and clinical traits
   - **Variants / Mutations**: Specific genetic variants (e.g., c.5A>G, p.Met1?)
4. Do NOT invent or assume any information not explicitly stated in the sources.
5. If a question is about something unrelated to RARS1 (e.g., diabetes, flu, hair color, bones),
   respond ONLY with "I do not know based on available literature." — nothing else.

SOURCES:
{context}

QUESTION: {query}

ANSWER:"""

        return self._call_llm_with_retry(prompt)

    def interactive_mode(self):
        print("\n" + "=" * 60)
        print("  RARS1 Genomic-RAG — Interactive Q&A Mode")
        print("  Type 'quit' or 'exit' to stop.")
        print("=" * 60 + "\n")

        while True:
            try:
                user_input = input("Your question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Interactive mode terminated.")
                break

            print("\n[Generating response...]\n")
            answer = self.ask(user_input)
            print(answer)
            print("\n" + "-" * 60 + "\n")

    def run_evaluation(self):
        print("\n" + "=" * 60)
        print("  Hallucination Guardrail Evaluation Starting")
        print(f"  Model: {MODEL_NAME} | Delay between calls: {REQUEST_DELAY}s")
        print("=" * 60)

        trick_cases = [
            {"query": "Does RARS1 mutation cause Type 1 Diabetes?", "expected": "negative"},
            {"query": "Is RARS1 associated with seasonal flu or influenza?", "expected": "negative"},
            {"query": "Does the RARS1 gene cause hair to turn blue?", "expected": "negative"},
            {"query": "Is RARS1 linked to broken bone recovery speed?", "expected": "negative"},
            {"query": "Does RARS1 cause cardiovascular disease or heart attacks?", "expected": "negative"},
        ]

        valid_cases = [
            {
                "query": "What are the most recently reported variants in RARS1 and their associated symptoms?",
                "expected": "positive"
            },
            {
                "query": "What neurological symptoms are associated with RARS1 mutations?",
                "expected": "positive"
            },
        ]

        all_cases = trick_cases + valid_cases
        results = []

        guardrail_keywords = [
            "do not know", "not found", "no evidence", "not mentioned",
            "not present", "cannot find", "not reported", "no information",
            "not related", "not associated", "not in the", "not supported"
        ]

        for idx, case in enumerate(all_cases):
            query = case["query"]
            expected = case["expected"]
            print(f"\n[TEST {idx + 1}/{len(all_cases)}] {query}")

            response = self.ask(query)

            # Check for error responses
            if response.startswith("[ERROR]"):
                print(f" API error: {response}")
                results.append({
                    "test_type": "Error",
                    "query": query,
                    "expected": expected,
                    "llm_response": response,
                    "contains_guardrail_phrase": False,
                    "passed": False
                })
                # Save partial results so far
                self._save_eval_results(results, partial=True)
                continue

            response_lower = response.lower()
            contains_guardrail = any(kw in response_lower for kw in guardrail_keywords)

            if expected == "negative":
                passed = contains_guardrail
                test_type = "Hallucination Guardrail"
            else:
                passed = not contains_guardrail
                test_type = "Valid Query Response"

            print(f"  [{test_type}] {'PASSED' if passed else 'FAILED'}")

            results.append({
                "test_type": test_type,
                "query": query,
                "expected": expected,
                "llm_response": response,
                "contains_guardrail_phrase": contains_guardrail,
                "passed": passed
            })

            self._save_eval_results(results, partial=(idx < len(all_cases) - 1))

            if idx < len(all_cases) - 1:
                print(f" Waiting {REQUEST_DELAY}s before next request...")
                time.sleep(REQUEST_DELAY)

        passed_count = sum(1 for r in results if r["passed"])
        total = len(results)
        print(f"\n  Evaluation complete: {passed_count}/{total} tests passed.")
        print(f"  Results saved to eval_results.json")
        print("=" * 60 + "\n")

    def _save_eval_results(self, results: list, partial: bool = False):
        """Saves current results to eval_results.json. Called after every test."""
        passed_count = sum(1 for r in results if r["passed"])
        total = len(results)

        eval_output = {
            "project": "RARS1 Genomic-RAG",
            "model": MODEL_NAME,
            "embedding": "S-BioBERT (pritamdeka/S-BioBert-snli-multinli-stsb)",
            "status": "partial" if partial else "complete",
            "summary": {
                "total_tests": total,
                "passed": passed_count,
                "failed": total - passed_count,
                "pass_rate": f"{(passed_count / total) * 100:.1f}%" if total > 0 else "N/A"
            },
            "tests": results
        }

        with open("eval_results.json", "w", encoding="utf-8") as f:
            json.dump(eval_output, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    rag = GenomicRAG()


    rag.setup()

 
    demo_query = "What are the most recently reported variants in RARS1 and their associated symptoms?"
    print(f"Demo Query: {demo_query}\n")
    print(rag.ask(demo_query))
    print("\n" + "=" * 60 + "\n")


    rag.run_evaluation()

    rag.interactive_mode()