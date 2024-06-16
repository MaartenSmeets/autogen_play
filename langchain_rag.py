import os
import logging
import re
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import chromadb

# Disable ChromaDB telemetry
os.environ["CHROMA_TELEMETRY"] = "FALSE"

# Configuration parameters
STORAGE_PATH = "./vector_store"
OLLAMA_API_KEY = 'Welcome01'
EMBEDDING_MODEL = "avr/sfr-embedding-mistral:latest"
LLM_MODEL = "mixtral:8x22b"
DOCUMENT_DIRECTORY = './crawled_pages'
QUESTION = "An Eldritch Knight (PHB fighter subclass) can choose level 3 spells when level 13. Which spell should I choose as an Eldritch Knight elf focused on ranged combat?"

# Logging configuration parameters
LOGGING_LEVEL = logging.INFO
TEXT_SNIPPET_LENGTH = 50
LOG_FILE = 'script_log.log'

# Configure logging
logging.basicConfig(level=LOGGING_LEVEL, format='%(name)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler(LOG_FILE),
    logging.StreamHandler()
])
logger = logging.getLogger(__name__)

# Reduce verbosity of other loggers
logging.getLogger("langchain_community.document_loaders.directory").setLevel(logging.WARNING)
logging.getLogger("unstructured").setLevel(logging.WARNING)
logging.getLogger("MARKDOWN").setLevel(logging.WARNING)

# Initialize OllamaEmbeddings
ollama_emb = OllamaEmbeddings(model=EMBEDDING_MODEL)

def get_embedding(texts, is_query=False):
    snippet = str(len(texts)) if isinstance(texts, list) else texts[:TEXT_SNIPPET_LENGTH]
    if is_query:
        logger.info(f"Embedding query texts: {snippet}...")
        return ollama_emb.embed_query(texts)
    logger.info(f"Embedding documents: {snippet}...")
    return ollama_emb.embed_documents(texts)

def clean_search_terms(search_terms):
    # Convert to lowercase
    search_terms = search_terms.lower()
    # Remove special characters and digits
    search_terms = re.sub(r'[^a-z\s]', '', search_terms)
    # Remove extra whitespace
    search_terms = re.sub(r'\s+', ' ', search_terms).strip()
    # Further refine search terms
    search_terms = re.sub(r'\bkeywords\b|\bsearch\b|\bterms\b', '', search_terms).strip()
    return search_terms

def generate_response(context, question, llm_model, task_type="answer"):
    context_snippet = str(len(context)) if isinstance(context, list) else context[:TEXT_SNIPPET_LENGTH]
    if task_type == "search_terms":
        prompt_template = "Determine the main keywords or search terms for the following question: {question}"
        input_data = {"question": question}
    elif task_type == "validate_document":
        prompt_template = "Context: {context}\n\nQuestion: {question}\n\nIs this document relevant to the question?"
        input_data = {"context": context, "question": question}
    else:
        prompt_template = "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        input_data = {"context": context, "question": question}
    
    logger.info(f"Generating {task_type} with context: '{context[:TEXT_SNIPPET_LENGTH]}' and question: '{question[:TEXT_SNIPPET_LENGTH]}'")

    ollama_llm = ChatOllama(model=llm_model)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | ollama_llm | StrOutputParser()

    logger.debug(f"Input data for prompt: {input_data}")

    try:
        response = chain.invoke(input_data)
        logger.debug(f"Response from Ollama: {response}")

        result_text = response
        logger.info(f"Generated {task_type} text: {result_text[:TEXT_SNIPPET_LENGTH]}...")
        return result_text
    except Exception as e:
        logger.error(f"Error generating {task_type}: {e}", exc_info=True)
        raise

def index_documents(directory):
    logger.info(f"Indexing documents in directory: {directory}")
    loader = DirectoryLoader(directory, glob="**/*.md")
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} documents")

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
    logger.info(f"Querying vector store with query: {query[:TEXT_SNIPPET_LENGTH]}...")
    embedding = get_embedding(query, is_query=True)
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    logger.info(f"Query results: {results}")
    return results

def validate_document_relevance(document, question, llm_model):
    response = generate_response(document, question, llm_model, task_type="validate_document")
    return "yes" in response.lower()

def main():
    collection = index_documents(DOCUMENT_DIRECTORY)

    search_terms = generate_response("", QUESTION, LLM_MODEL, task_type="search_terms")
    logger.info(f"Determined search terms: {search_terms[:TEXT_SNIPPET_LENGTH]}")

    cleaned_search_terms = clean_search_terms(search_terms)
    logger.info(f"Cleaned search terms: {cleaned_search_terms}")

    results = query_vector_store(collection, cleaned_search_terms, top_k=1)
    logger.info(f"Query results: {results}")

    relevant_documents = []
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

if __name__ == "__main__":
    main()
