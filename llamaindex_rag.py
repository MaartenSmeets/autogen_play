import logging
from openai import OpenAI
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.prompts import ChatPromptTemplate
import chromadb
from bs4 import BeautifulSoup
import math
import uuid

# Base URL for the local OpenAI-compliant API and Ollama
BASE_URL = "http://localhost:11434"
API_KEY = "Welcome01"
# Configuration parameters
STORAGE_PATH = "./vector_store"
EMBEDDING_MODEL = "nomic-embed-text:latest"  # Specify your embedding model
LLM_MODEL = "mixtral:8x22b-instruct-v0.1-q3_K_S"  # Updated model
DOCUMENT_DIRECTORY = './crawled_pages'
QUESTION = "An Eldritch Knight (PHB fighter subclass) can choose level 3 spells when level 13. Which spell should I choose as an Eldritch Knight elf focused on ranged combat?"
UPDATE_VECTOR_STORE = False  # New parameter to make updating the vector store optional

# Logging configuration parameters
LOGGING_LEVEL = logging.INFO
TEXT_SNIPPET_LENGTH = 200
LOG_FILE = 'script_log.log'

# Token length parameter
TOKEN_LENGTH = 4096

# Configure logging
logging.basicConfig(level=LOGGING_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', handlers=[
    logging.FileHandler(LOG_FILE),
    logging.StreamHandler()
])
logger = logging.getLogger(__name__)

# Reduce verbosity of other loggers
logging.getLogger("llama_index.readers").setLevel(logging.WARNING)
logging.getLogger("llama_index.node_parser").setLevel(logging.WARNING)

# Initialize OllamaEmbedding with the local URL and specified model
ollama_emb = OllamaEmbedding(model_name=EMBEDDING_MODEL, base_url=BASE_URL)

def get_embedding(texts, is_query=False):
    snippet = str(len(texts)) if isinstance(texts, list) else texts[:TEXT_SNIPPET_LENGTH]
    if is_query:
        logger.info(f"Embedding query texts: {snippet}...")
        return ollama_emb.get_query_embedding(texts)

    logger.info(f"Embedding documents: {snippet}...")
    return ollama_emb.get_text_embedding_batch(texts, show_progress=True)

def generate_response(context, question, llm_model, task_type="answer"):
    prompt_template = ""

    if task_type == "search_terms":
        prompt_template = "Provide only a comma-separated list of main keywords for a semantic search based on the following question. Avoid an introductory polite sentence. Keep the result on a single line. Do not use spaces before or after a comma. Mind capitals. Mind to put terms which belong together as the same keyword. Do not replace spaces with _. Mind that very general keywords will not yield usable results so be specific: {question}"
    elif task_type == "validate_document":
        prompt_template = "Context: {context}\n\nQuestion: {question}\n\nWill this document be of added value in answering this question? Please answer only with 'yes' or 'no'."
    else:
        prompt_template = "Context: {context}\n\nQuestion: {question}\n\nAnswer:"

    logger.info(f"Generating {task_type} with context: '{context[:TEXT_SNIPPET_LENGTH]}' and question: '{question[:TEXT_SNIPPET_LENGTH]}'")

    prompt = prompt_template.format(context=context, question=question)
    try:
        client = OpenAI(base_url=(BASE_URL+"/v1/"), api_key=API_KEY)
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=TOKEN_LENGTH,
            temperature=0.1
        )
        logger.debug(f"Response from OpenAI: {response}")

        result_text = response.choices[0].message['content'].strip()
        logger.info(f"Generated {task_type} text: {result_text[:TEXT_SNIPPET_LENGTH]}...")        
        return result_text
    except Exception as e:
        logger.error(f"Error generating {task_type}: {e}", exc_info=True)
        raise

def read_html_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    soup = BeautifulSoup(content, 'html.parser')
    return soup.get_text()

def index_documents(directory):
    if UPDATE_VECTOR_STORE:
        logger.info(f"Indexing documents in directory: {directory}")
        loader = SimpleDirectoryReader(directory)
        documents = loader.load_data()

        logger.info(f"Loaded {len(documents)} documents")

        # Semantic Chunking
        embed_model = ollama_emb
        logger.info("Start splitting documents into semantic chunks")
        splitter = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model)
        nodes = []

        total_documents = len(documents)
        last_logged_progress = 0

        # Process each document and update progress
        for i, document in enumerate(documents):
            nodes.extend(splitter.get_nodes_from_documents([document]))

            # Log progress every 5%
            current_progress = (i + 1) / total_documents * 100
            if current_progress - last_logged_progress >= 5:
                logger.info(f"Progress: {current_progress:.2f}%")
                last_logged_progress = current_progress

        total_nodes = len(nodes)
        logger.info(f"Number of chunks determined for vector store: {total_nodes}")

        client = chromadb.PersistentClient(path=STORAGE_PATH)
        collection = client.get_or_create_collection(name="documents")

        if collection.count() == 0:
            texts = [node.get_content() for node in nodes]
            embeddings = get_embedding(texts)
            metadatas = [{"source": node.metadata.get('source', '')} for node in nodes]
            ids = [{uuid.uuid4()} for node in nodes]

            logger.info(f"Adding documents to collection with {len(texts)} texts")
            collection.add(documents=texts, metadatas=metadatas, embeddings=embeddings, ids=ids)

        return collection
    else:
        logger.info("Skipping document indexing as UPDATE_VECTOR_STORE is set to False.")
        return None

def query_vector_store(collection, query, top_k=5):
    logger.info(f"Querying vector store with query: {query[:TEXT_SNIPPET_LENGTH]}...")
    embedding = get_embedding(query, is_query=True)
    results = collection.query(query_embeddings=[embedding], n_results=top_k)

    if top_k > 1:
        logger.info(f"Top {top_k} results:")
        for idx, doc in enumerate(results['documents'], start=1):
            logger.info(f"Result {idx}: {doc[0][:TEXT_SNIPPET_LENGTH]}...")

    return results

def validate_document_relevance(document, question, llm_model):
    response = generate_response(document, question, llm_model, task_type="validate_document")
    return response.strip().lower() == 'yes'

def main():
    collection = None
    if UPDATE_VECTOR_STORE:
        collection = index_documents(DOCUMENT_DIRECTORY)
    else:
        logger.info("Skipping document indexing as UPDATE_VECTOR_STORE is set to False.")
        client = chromadb.PersistentClient(path=STORAGE_PATH)
        collection = client.get_collection(name="documents")

    search_terms = generate_response("", QUESTION, LLM_MODEL, task_type="search_terms")
    logger.info(f"Determined search terms: {search_terms}")

    # Split search terms into individual terms
    search_terms_list = search_terms.split(',')
    logger.info(f"Search terms list: {search_terms_list}")

    if collection:
        relevant_documents = []
        for term in search_terms_list:
            results = query_vector_store(collection, term, top_k=10)

            if results and 'documents' in results:
                for doc in results['documents']:
                    if validate_document_relevance(doc[0], QUESTION, LLM_MODEL):
                        relevant_documents.append(doc[0])

        if relevant_documents:
            context = " ".join([doc[:TEXT_SNIPPET_LENGTH] for doc in relevant_documents])
            logger.info(f"Generated context: {context[:TEXT_SNIPPET_LENGTH]}...")
            answer = generate_response(context, QUESTION, LLM_MODEL, task_type="answer")
            logger.info(f"Question: {QUESTION}\nAnswer: {answer}")
        else:
            logger.info("No relevant documents found.")
    else:
        logger.info("Collection is not available.")

if __name__ == "__main__":
    main()
    
