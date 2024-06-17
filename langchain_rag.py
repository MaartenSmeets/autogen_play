import os
import logging
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import chromadb

# Disable ChromaDB telemetry
os.environ["CHROMA_TELEMETRY"] = "False"

# Configuration parameters
STORAGE_PATH = "./vector_store"
OLLAMA_API_KEY = 'Welcome01'
EMBEDDING_MODEL = "avr/sfr-embedding-mistral:latest"
LLM_MODEL = "mixtral:8x22b-instruct-v0.1-q3_K_S"
DOCUMENT_DIRECTORY = './crawled_pages'
QUESTION = "An Eldritch Knight (PHB fighter subclass) can choose level 3 spells when level 13. Which spell should I choose as an Eldritch Knight elf focused on ranged combat?"
UPDATE_VECTOR_STORE = True  # New parameter to make updating the vector store optional

# Logging configuration parameters
LOGGING_LEVEL = logging.INFO
TEXT_SNIPPET_LENGTH = 200
LOG_FILE = 'script_log.log'

# Token length parameter
TOKEN_LENGTH = 4096

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

def generate_response(context, question, llm_model, task_type="answer"):
    if task_type == "search_terms":
        prompt_template = "Provide only a comma-separated list of main keywords for a semantic search based on the following question. Avoid an introductionary polite sentence. Keep the result on a single line. Do not use spaces before or after a comma. Mind capitals. Mind to put terms which belong together as the same keyword. Do not replace spaces with _. Mind that very general keywords will not yield usable results so be specific: {question}"
        input_data = {"question": question}
    elif task_type == "validate_document":
        prompt_template = "Context: {context}\n\nQuestion: {question}\n\nWill this document be of added value in answering this question? Please answer only with 'yes' or 'no'."
        input_data = {"context": context, "question": question}
    else:
        prompt_template = "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        input_data = {"context": context, "question": question}
    
    logger.info(f"Generating {task_type} with context: '{context[:TEXT_SNIPPET_LENGTH]}' and question: '{question[:TEXT_SNIPPET_LENGTH]}'")

    ollama_llm = ChatOllama(model=llm_model, token_limit=TOKEN_LENGTH)
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
    if UPDATE_VECTOR_STORE:
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
    first_line = response.split('\n')[0]
    return first_line.strip().lower().startswith('yes')

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
