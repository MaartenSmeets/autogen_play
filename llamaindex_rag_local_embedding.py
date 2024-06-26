import logging
from openai import OpenAI
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import Document
import chromadb
from bs4 import BeautifulSoup
import uuid
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuration parameters
BASE_URL = "http://localhost:11434"
API_KEY = "Welcome01"
STORAGE_PATH = "./vector_store"
TITLE_GENERATION_MODEL = "mistral:instruct"
EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
LLM_MODEL = "mixtral:8x22b-instruct-v0.1-q3_K_S"
DOCUMENT_DIRECTORY = './crawled_pages'
QUESTION = "Which level 3 spells would be useful for a DND 5th edition eldritch knight elf focused on ranged combat?"
LOGGING_LEVEL = logging.INFO
TEXT_SNIPPET_LENGTH = 200
LOG_FILE = 'script_log.log'
TOKEN_LENGTH = 16384
PROCESSING_BATCH_SIZE = 200
USE_LLM_FOR_TITLE = True

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

# Initialize embedding model
embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)

def transform_query(query: str) -> str:
    """ For retrieval, add the prompt for query (not for documents).
    """
    return f'Represent this sentence for searching relevant passages: {query}'

def get_query_embedding(query):
    transformed_query = transform_query(query)
    logger.debug("Embedding query...")
    return embed_model.get_query_embedding(transformed_query)

def get_text_embeddings(texts):
    logger.debug("Embedding documents...")
    return embed_model.get_text_embedding_batch(texts, show_progress=True)

def generate_response(contexts, questions, llm_model, task_type="answer"):
    prompts = {
        "answer": "Context: {context}\n\nQuestion: {question}\n\nAnswer:",
        "generate_title": "Context: {context}\n\nGenerate only a concise title for the above context.",
        "validate_document": "Context: {context}\n\nQuestion: {question}\n\nIs this document relevant and useful in answering the question? Answer 'yes' or 'no' with a brief justification.",
    }
    prompt_template = prompts.get(task_type, prompts["answer"])
    
    client = OpenAI(base_url=(BASE_URL + "/v1/"), api_key=API_KEY)
    responses = []
    
    for context, question in zip(contexts, questions):
        prompt = prompt_template.format(context=context, question=question)
        logger.debug(f"Generating {task_type} with question: '{question[:TEXT_SNIPPET_LENGTH]}'")

        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=TOKEN_LENGTH,
                temperature=0.1
            )
            logger.debug(f"Full response received")
            result_text = response.choices[0].message.content.strip()
            logger.debug(f"Generated {task_type} text...")
            
            if task_type == "generate_title":
                result_text = result_text.split('\n')[-1].strip()
                if result_text.startswith("Title: "):
                    result_text = result_text[len("Title: "):].strip()
                # Remove surrounding quotes if present
                result_text = result_text.strip('"\'')
            
            responses.append(result_text)
        except Exception as e:
            logger.error(f"Error generating {task_type}: {e}", exc_info=True)
            raise

    return responses

def read_html_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        soup = BeautifulSoup(content, 'html.parser')
        soup_text = soup.get_text()
        title = soup.title.string if soup.title else "No title"
        logger.debug(f"Read HTML file: {filepath} with title: {title}")
        return soup_text, title
    except Exception as e:
        logger.error(f"Error reading HTML file {filepath}: {e}", exc_info=True)
        raise

def preprocess_text(text):
    lines = text.splitlines()
    seen_lines = set()
    filtered_lines = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line == "":
            continue
        if stripped_line in seen_lines:
            continue
        seen_lines.add(stripped_line)
        filtered_lines.append(line)
    return "\n".join(filtered_lines).strip()

def index_document_batch(documents, splitter, collection):
    sources = [doc.metadata.get('source', None) for doc in documents]
    contexts = []
    html_titles = []
    llm_titles = []
    
    for document, source in zip(documents, sources):
        if not source:
            logger.error("Document has no source metadata.")
            continue
        
        logger.debug(f"Processing file: {source}")
        file_path = os.path.join(DOCUMENT_DIRECTORY, source)
        if source.endswith('.html'):
            logger.debug(f"File identified as HTML: {source}")
            document_content, document_title = read_html_file(file_path)
            document_content = preprocess_text(document_content)
            document.text = document_content
            html_titles.append(document_title)
        else:
            document.text = preprocess_text(document.text)
            logger.debug(f"File not identified as HTML: {source}")
            html_titles.append("")

        contexts.append(document.text)
    
    # Generate titles if needed
    if USE_LLM_FOR_TITLE:
        generated_llm_titles = generate_response(contexts, [""] * len(contexts), TITLE_GENERATION_MODEL, task_type="generate_title")
        llm_titles = generated_llm_titles
    
    for document, html_title, llm_title in zip(documents, html_titles, llm_titles):
        document.metadata['html_title'] = html_title
        document.metadata['llm_title'] = llm_title

    # Split documents into chunks and get embeddings
    all_chunks = []
    for document in documents:
        chunks = splitter.get_nodes_from_documents([document])
        for chunk in chunks:
            chunk.metadata['html_title'] = document.metadata['html_title']
            chunk.metadata['llm_title'] = document.metadata['llm_title']
            chunk.metadata['source'] = document.metadata['source']
        all_chunks.extend(chunks)

    texts = [chunk.get_content() for chunk in all_chunks]
    embeddings = get_text_embeddings(texts)
    metadatas = [{"source": chunk.metadata.get('source', ''), "html_title": chunk.metadata.get('html_title', ''), "llm_title": chunk.metadata.get('llm_title', '')} for chunk in all_chunks]
    
    # Improved ID generation using Composite ID
    ids = [f"{chunk.metadata['source']}-{i}-{uuid.uuid4()}" for i, chunk in enumerate(all_chunks)]
    
    collection.add(documents=texts, metadatas=metadatas, embeddings=embeddings, ids=ids)

    for source in sources:
        logger.info(f"Processed and indexed document: {source}")

def index_documents(directory):
    client = chromadb.PersistentClient(path=STORAGE_PATH)
    try:
        collection = client.get_collection(name="documents")
        logger.info("Collection 'documents' already exists. Skipping indexing.")
    except ValueError:
        logger.info(f"Collection 'documents' does not exist. Indexing documents in directory: {directory}")
        collection = client.create_collection(name="documents")

        # Best practices for using SemanticSplitterNodeParser
        splitter = SemanticSplitterNodeParser(
            buffer_size=3,  # Adjust buffer size to include more contextual information
            embed_model=embed_model,  # Using the embedding model initialized earlier
        )
        
        documents = []
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                metadata = {'source': os.path.relpath(file_path, directory)}
                document = Document(text=content, metadata=metadata)
                documents.append(document)
        
        # Process documents in batches
        batch_size = PROCESSING_BATCH_SIZE
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            index_document_batch(batch, splitter, collection)

    return collection

def query_vector_store(collection, query, top_k=5):
    logger.info(f"Querying vector store with query: {query[:TEXT_SNIPPET_LENGTH]}...")
    embedding = get_query_embedding(query)
    results = collection.query(query_embeddings=[embedding], n_results=top_k, include=["documents", "metadatas"])
    return results

def validate_document_relevance(document, question, llm_model):
    response = generate_response([document], [question], llm_model, task_type="validate_document")
    response_text = response[0].strip().lower()
    justification = response_text.split(':')[1].strip() if ':' in response_text else ''
    logger.info(f"Validating document relevance: {response_text}")
    return response_text.startswith('yes')

def main():
    collection = index_documents(DOCUMENT_DIRECTORY)

    if collection is None:
        logger.error("Failed to create or retrieve the collection. Exiting.")
        return

    logger.info(f"Using direct query for vector store search: {QUESTION}")

    if collection:
        all_relevant_documents = []
        all_irrelevant_documents = []
        results = query_vector_store(collection, QUESTION, top_k=10)
        if results and 'documents' in results and 'metadatas' in results:
            for docs, metas in zip(results['documents'], results['metadatas']):
                for doc, meta in zip(docs, metas):
                    if validate_document_relevance(doc, QUESTION, LLM_MODEL):
                        logger.info(f"Found relevant document with HTML title: '{meta.get('html_title', 'No title')}' and LLM title: '{meta.get('llm_title', 'No title')}'")
                        all_relevant_documents.append((doc, meta))
                    else:
                        logger.info(f"Found irrelevant document with HTML title: '{meta.get('html_title', 'No title')}' and LLM title: '{meta.get('llm_title', 'No title')}'")
                        all_irrelevant_documents.append((doc, meta))

        if all_relevant_documents:
            context = " ".join([doc for doc, _ in all_relevant_documents])
            logger.info(f"Generated context...")
            answer = generate_response([context], [QUESTION], LLM_MODEL, task_type="answer")
            logger.info(f"Question: {QUESTION}\nAnswer: {answer[0]}")
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
