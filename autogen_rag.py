"""
This module contains code for autogenerating responses using RetrieveAssistantAgent and RetrieveUserProxyAgent.
"""

import autogen.retrieve_utils
from chromadb.utils import embedding_functions
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from langchain.text_splitter import RecursiveCharacterTextSplitter
import autogen
import os


config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        #"model": ["llama3:70b"]
        "model": ["mixtral:8x22b"]
    },
)

llm_config = {
    "timeout": 7200,  # timeout for each request to OpenAI API
    "max_retries": 10,  # maximum number of retries for each request to OpenAI API
    "config_list": config_list,  # a list of OpenAI API configurations
    "max_tokens": 65536,  # maximum number of tokens for each response
    "temperature": 0.2,  # temperature for sampling
    "cache_seed": None
}

text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\r", "\t"])

assistant = RetrieveAssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config=llm_config,
)

ef = embedding_functions.OllamaEmbeddingFunction(
    model_name="nomic-embed-text:latest",
    url="http://localhost:11434/api/embeddings"
)

ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    retrieve_config={
        "task": "qa",
        "docs_path": ["./crawled_pages"],
        "embedding_function": embedding_functions.DefaultEmbeddingFunction(),
        "collection_name": "rag_collection",
        "vector_db": "chroma",
        "custom_text_split_function": text_splitter.split_text,
        "human_input_mode": "NEVER",
        "overwrite": False
    },
    code_execution_config=False,
)
assistant.reset()
ragproxyagent.initiate_chat(
    assistant,
    message=ragproxyagent.message_generator,
    problem="Which level 3 spells can an Eldritch Knight (fighter subclass) choose from and why? Consider the allowed spell schools"
)
