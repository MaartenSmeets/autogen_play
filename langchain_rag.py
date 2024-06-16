import os
import logging
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
import chromadb

# Configuration parameters
STORAGE_PATH = "./vector_store"  # Path to store the vector store
OLLAMA_API_KEY = 'Welcome01'
EMBEDDING_MODEL = "qwen2:7b-instruct"
LLM_MODEL = "mixtral:8x22b"  # Specify your LLM model here
DOCUMENT_DIRECTORY = './crawled_pages'
QUESTION = "An Eldritch Knight (PHB fighter subclass) can choose level 3 spells when level 13. Which spell should I choose as an Eldritch Knight elf focused on ranged combat?"

# Logging configuration parameters
LOGGING_LEVEL = logging.INFO  # Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
DEBUG_LOGGING_LEVEL = logging.DEBUG
CONSOLE_LOGGING_LEVEL = logging.DEBUG
TEXT_SNIPPET_LENGTH = 50  # Number of characters to show in log snippets

# Configure logging
logging.basicConfig(level=LOGGING_LEVEL)
logger = logging.getLogger(__name__)

debug_logger = logging.getLogger("debug_logger")
debug_logger.setLevel(DEBUG_LOGGING_LEVEL)

console_handler = logging.StreamHandler()
console_handler.setLevel(CONSOLE_LOGGING_LEVEL)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
debug_logger.addHandler(console_handler)

# Reduce verbosity of other loggers
logging.getLogger("langchain_community.document_loaders.directory").setLevel(logging.WARNING)
logging.getLogger("unstructured").setLevel(logging.WARNING)
logging.getLogger("MARKDOWN").setLevel(logging.WARNING)

# Initialize OllamaEmbeddings
ollama_emb = OllamaEmbeddings(model=EMBEDDING_MODEL)

def get_embedding(texts, is_query=False):
    snippet = texts[:TEXT_SNIPPET_LENGTH]
    if is_query:
        debug_logger.debug(f"Embedding query texts: {snippet}...")
        return ollama_emb.embed_query(texts)
    debug_logger.debug(f"Embedding documents: {snippet}...")
    return ollama_emb.embed_documents(texts)

def generate_answer(context, question, llm_model):
    context_snippet = context[:TEXT_SNIPPET_LENGTH]
    debug_logger.debug(f"Generating answer with context: {context_snippet}... and question: {question}")
    ollama_llm = ChatOllama(model=llm_model)
    messages = [HumanMessage(content=f"Context: {context}\n\nQuestion: {question}\n\nAnswer:")]

    try:
        response = ollama_llm.generate(messages, max_tokens=150, temperature=0.1)
        debug_logger.debug(f"Generated response: {response}")

        if isinstance(response.generations, list) and response.generations:
            answer_text = " ".join([gen.text for gen in response.generations])
        else:
            answer_text = response.generations[0].text
        
        debug_logger.debug(f"Generated answer text: {answer_text[:TEXT_SNIPPET_LENGTH]}...")
        return answer_text
    except Exception as e:
        debug_logger.error(f"Error generating answer: {e}")
        raise

def determine_search_terms(question, llm_model):
    debug_logger.debug(f"Determining search terms for question: {question}")
    ollama_llm = ChatOllama(model=llm_model)
    messages = [HumanMessage(content=f"Determine the main keywords or search terms for the following question to use for a semantic search: {question}")]

    try:
        response = ollama_llm.generate(messages, max_tokens=50, temperature=0.1)
        debug_logger.debug(f"Generated response: {response}")

        if isinstance(response.generations, list) and response.generations:
            search_terms = " ".join([gen.text for gen in response.generations])
        else:
            search_terms = response.generations[0].text
        
        debug_logger.debug(f"Determined search terms: {search_terms[:TEXT_SNIPPET_LENGTH]}")
        return search_terms
    except Exception as e:
        debug_logger.error(f"Error determining search terms: {e}")
        raise

def index_documents(directory):
    logger.info(f"Indexing documents in directory: {directory}")
    loader = DirectoryLoader(directory, glob="**/*.md")  # Assuming documents are markdown files
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} documents")
    
    # Initialize Chroma vector store with persistent client
    client = chromadb.PersistentClient(path=STORAGE_PATH)
    collection = client.get_or_create_collection(name="documents")

    if collection.count() == 0:
        texts = [doc.page_content for doc in documents]
        embeddings = get_embedding(texts)
        metadatas = [{"source": doc.metadata.get('source', '')[:TEXT_SNIPPET_LENGTH]} for doc in documents]
        ids = [doc.metadata.get('source', '')[:TEXT_SNIPPET_LENGTH] for doc in documents]

        logger.info(f"Adding documents to collection with {len(texts)} texts")
        collection.add(documents=texts, metadatas=metadatas, embeddings=embeddings, ids=ids)
    
    return collection

def query_vector_store(collection, query, top_k=5):
    logger.info(f"Querying vector store with query: {query}")
    embedding = get_embedding(query, is_query=True)
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    logger.info(f"Query results: {results}")
    return results

def main():
    collection = index_documents(DOCUMENT_DIRECTORY)
    
    # Determine the search terms for the question
    search_terms = determine_search_terms(QUESTION, LLM_MODEL)
    logger.info(f"Determined search terms: {search_terms[:TEXT_SNIPPET_LENGTH]}")
    
    # Use the search terms to query the vector store
    results = query_vector_store(collection, search_terms, top_k=1)

    # Debugging: Print results to see the format
    logger.info(f"Query results: {results}")

    if results and 'documents' in results:
        # Flatten the list of lists
        context = " ".join([doc[:TEXT_SNIPPET_LENGTH] for sublist in results['documents'] for doc in sublist])
        logger.info(f"Generated context: {context[:TEXT_SNIPPET_LENGTH]}...")
        answer = generate_answer(context, QUESTION, LLM_MODEL)
        logger.info(f"Question: {QUESTION}\nAnswer: {answer}")
    else:
        logger.info("No relevant documents found.")

if __name__ == "__main__":
    main()
