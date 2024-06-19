import logging
from openai import OpenAI
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.schema import Document
import chromadb
from bs4 import BeautifulSoup
import uuid
import os

# Configuration parameters
BASE_URL = "http://localhost:11434"
API_KEY = "Welcome01"
STORAGE_PATH = "./vector_store"
EMBEDDING_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "mixtral:8x22b-instruct-v0.1-q3_K_S"
DOCUMENT_DIRECTORY = './crawled_pages'
QUESTION = "An Eldritch Knight (PHB fighter subclass) can choose level 3 spells when level 13. Which spell should I choose as an Eldritch Knight elf focused on ranged combat?"
UPDATE_VECTOR_STORE = False
LOGGING_LEVEL = logging.INFO
TEXT_SNIPPET_LENGTH = 200
LOG_FILE = 'script_log.log'
TOKEN_LENGTH = 16384

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)

# Create handlers
file_handler = logging.FileHandler(LOG_FILE, mode='w')
stream_handler = logging.StreamHandler()

# Set logging level for handlers
file_handler.setLevel(LOGGING_LEVEL)
stream_handler.setLevel(LOGGING_LEVEL)

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Reduce verbosity of other loggers
logging.getLogger("llama_index.readers").setLevel(logging.WARNING)
logging.getLogger("llama_index.node_parser").setLevel(logging.WARNING)

# Initialize OllamaEmbedding
ollama_emb = OllamaEmbedding(model_name=EMBEDDING_MODEL, base_url=BASE_URL)

def get_embedding(texts, is_query=False):
    snippet = str(len(texts)) if isinstance(texts, list) else texts[:TEXT_SNIPPET_LENGTH]
    logger.info(f"Embedding {'query' if is_query else 'documents'}...")
    return ollama_emb.get_query_embedding(texts) if is_query else ollama_emb.get_text_embedding_batch(texts, show_progress=True)

def generate_response(context, question, llm_model, task_type="answer"):
    prompts = {
        "search_terms": "Provide only a comma-separated list of main keywords for a semantic search based on the following question: {question}",
        "validate_document": "Context: {context}\n\nQuestion: {question}\n\nWill this document be of added value in answering this question? Please answer only with 'yes' or 'no'.",
        "answer": "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    }
    prompt_template = prompts.get(task_type, prompts["answer"])
    prompt = prompt_template.format(context=context, question=question)
    logger.info(f"Generating {task_type} with question: '{question[:TEXT_SNIPPET_LENGTH]}'")

    try:
        client = OpenAI(base_url=(BASE_URL + "/v1/"), api_key=API_KEY)
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=TOKEN_LENGTH,
            temperature=0.1
        )
        logger.info(f"Full response received")

        result_text = response.choices[0].message.content.strip()
        logger.info(f"Generated {task_type} text...")
        return result_text
    except Exception as e:
        logger.error(f"Error generating {task_type}: {e}", exc_info=True)
        raise

def read_html_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        soup_text = BeautifulSoup(content, 'html.parser').get_text()
        logger.debug(f"Read HTML file: {filepath}")
        return soup_text
    except Exception as e:
        logger.error(f"Error reading HTML file {filepath}: {e}", exc_info=True)
        raise

def preprocess_text(text):
    lines = text.splitlines()
    filtered_lines = [line for line in lines if line.strip() != '']
    return "\n".join(filtered_lines).strip()

def index_documents(directory):
    if UPDATE_VECTOR_STORE:
        logger.info(f"Indexing documents in directory: {directory}")

        # Manually load files and assign metadata
        documents = []
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                metadata = {'source': os.path.relpath(file_path, directory)}
                documents.append(Document(text=content, metadata=metadata))

        logger.info(f"Loaded {len(documents)} documents")

        splitter = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95, embed_model=ollama_emb)
        nodes = []
        total_documents = len(documents)

        for i, document in enumerate(documents):
            source = document.metadata.get('source', None)
            if not source:
                logger.error(f"Document {i} has no source metadata.")
                continue

            file_path = os.path.join(directory, source)
            if source.endswith('.html'):
                logger.debug(f"File identified as HTML: {source}")
                document_content = read_html_file(file_path)
                document_content = preprocess_text(document_content)
                document = Document(text=document_content, metadata={'source': source})
            else:
                document.text = preprocess_text(document.text)
                logger.debug(f"File not identified as HTML: {source}")

            nodes.extend(splitter.get_nodes_from_documents([document]))
            current_progress = (i + 1) / total_documents * 100
            if (i + 1) % max(1, total_documents // 20) == 0:  # Log every 5%
                logger.info(f"Progress splitting documents in semantic chunks: {current_progress:.2f}%")

        if not nodes:
            logger.error("No nodes were created. Please check the input documents and metadata.")
            return None

        client = chromadb.PersistentClient(path=STORAGE_PATH)
        collection = client.get_or_create_collection(name="documents")

        if collection.count() == 0:
            texts = [node.get_content() for node in nodes]
            embeddings = get_embedding(texts)
            metadatas = [{"source": node.metadata.get('source', '')} for node in nodes]
            ids = [str(uuid.uuid4()) for _ in nodes]
            collection.add(documents=texts, metadatas=metadatas, embeddings=embeddings, ids=ids)

        return collection
    else:
        logger.info("Skipping document indexing as UPDATE_VECTOR_STORE is set to False.")
        client = chromadb.PersistentClient(path=STORAGE_PATH)
        return client.get_collection(name="documents")

def query_vector_store(collection, query, top_k=5):
    logger.info(f"Querying vector store with query: {query[:TEXT_SNIPPET_LENGTH]}...")
    embedding = get_embedding(query, is_query=True)
    results = collection.query(query_embeddings=[embedding], n_results=top_k, include=["documents", "metadatas"])

    logger.debug(f"Query results structure: {results}")
    if top_k > 1:
        logger.info(f"Top {top_k} results:")
        for idx, (docs, metas) in enumerate(zip(results['documents'], results['metadatas']), start=1):
            for doc_idx, (doc, meta) in enumerate(zip(docs, metas)):
                source = meta.get('source', 'Unknown source')
                logger.info(f"Result {idx}, Document {doc_idx} from {source}")

    return results

def validate_document_relevance(document, question, llm_model):
    response = generate_response(document, question, llm_model, task_type="validate_document")
    return response.strip().lower() == 'yes'

def main():
    collection = index_documents(DOCUMENT_DIRECTORY)

    if collection is None:
        logger.error("Failed to create or retrieve the collection. Exiting.")
        return

    search_terms = generate_response("", QUESTION, LLM_MODEL, task_type="search_terms")
    logger.info(f"Determined search terms: {search_terms}")

    if collection:
        results = query_vector_store(collection, search_terms, top_k=10)
        relevant_documents = []
        
        if results and 'documents' in results:
            for docs, metas in zip(results['documents'], results['metadatas']):
                for doc, meta in zip(docs, metas):
                    if 'content' in meta and validate_document_relevance(doc, QUESTION, LLM_MODEL):
                        relevant_documents.append(doc)

        if relevant_documents:
            context = " ".join([doc for doc in relevant_documents])
            logger.info(f"Generated context...")
            answer = generate_response(context, QUESTION, LLM_MODEL, task_type="answer")
            logger.info(f"Question: {QUESTION}\nAnswer: {answer}")
        else:
            logger.info("No relevant documents found.")
    else:
        logger.info("Collection is not available.")

if __name__ == "__main__":
    try:
        main()
    finally:
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)
