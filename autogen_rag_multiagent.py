"""
This module contains code for autogenerating responses using RetrieveAssistantAgent and RetrieveUserProxyAgent.
"""

from typing import Annotated
import autogen.retrieve_utils
from chromadb.utils import embedding_functions
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from langchain.text_splitter import RecursiveCharacterTextSplitter
import autogen
import os


PROBLEM="Which level 3 spells can an Eldritch Knight (fighter subclass) choose from and why? Consider the allowed spell schools"

def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()


config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        #"model": ["mixtral:8x22b"]
        "model": ["llama3:70b"]
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
    #model_name="hellord/e5-mistral-7b-instruct:Q8_0",
    url="http://localhost:11434/api/embeddings"
)


boss = autogen.UserProxyAgent(
    name="Boss",
    is_termination_msg=termination_msg,
    human_input_mode="TERMINATE",
    system_message="The boss who ask questions and give tasks.",
)

boss_aid = RetrieveUserProxyAgent(
    name="Boss_Assistant",
    is_termination_msg=termination_msg,
    system_message="Assistant who has extra content retrieval power for solving difficult problems.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "qa",
        "docs_path": ["./crawled_pages"],
        "embedding_function": ef,
        "collection_name": "rag_collection",
        "vector_db": "chroma",
        "custom_text_split_function": text_splitter.split_text,
        "overwrite": False
    },
    code_execution_config=False,  # we don't want to execute code in this case.
)

coder = autogen.AssistantAgent(
    name="Senior_Python_Engineer",
    is_termination_msg=termination_msg,
    system_message="You are a senior python engineer. Reply `TERMINATE` in the end when everything is done.",
    llm_config=llm_config
)

pm = autogen.AssistantAgent(
    name="Product_Manager",
    is_termination_msg=termination_msg,
    system_message="You are a product manager. Reply `TERMINATE` in the end when everything is done.",
    llm_config=llm_config
)

reviewer = autogen.AssistantAgent(
    name="Code_Reviewer",
    is_termination_msg=termination_msg,
    system_message="You are a code reviewer. Reply `TERMINATE` in the end when everything is done.",
    llm_config=llm_config
)

def retrieve_content(
    message: Annotated[
        str,
        "Refined message which keeps the original meaning and can be used to retrieve content for code generation and question answering.",
    ],
    n_results: Annotated[int, "number of results"] = 3,
) -> str:
    boss_aid.n_results = n_results  # Set the number of results to be retrieved.
    # Check if we need to update the context.
    update_context_case1, update_context_case2 = boss_aid._check_update_context(message)
    if (update_context_case1 or update_context_case2) and boss_aid.update_context:
        boss_aid.problem = message if not hasattr(boss_aid, "problem") else boss_aid.problem
        _, ret_msg = boss_aid._generate_retrieve_user_reply(message)
    else:
        _context = {"problem": message, "n_results": n_results}
        ret_msg = boss_aid.message_generator(boss_aid, None, _context)
    return ret_msg if ret_msg else message

for caller in [pm, coder, reviewer]:
    d_retrieve_content = caller.register_for_llm(
        description="retrieve content for code generation and question answering.", api_style="function"
    )(retrieve_content)

for executor in [boss, pm]:
    executor.register_for_execution()(d_retrieve_content)

groupchat = autogen.GroupChat(
    agents=[boss, pm, coder, reviewer],
    messages=[],
    max_round=12,
    speaker_selection_method="round_robin",
    allow_repeat_speaker=False,
)

def _reset_agents():
    boss.reset()
    boss_aid.reset()
    coder.reset()
    pm.reset()
    reviewer.reset()


def function_calling_rag_chat():
    _reset_agents()
    def retrieve_content(message, n_results=3):
        boss_aid.n_results = n_results  # Set the number of results to be retrieved.
        # Check if we need to update the context.
        update_context_case1, update_context_case2 = boss_aid._check_update_context(message)
        if (update_context_case1 or update_context_case2) and boss_aid.update_context:
            boss_aid.problem = message if not hasattr(boss_aid, "problem") else boss_aid.problem
            _, ret_msg = boss_aid._generate_retrieve_user_reply(message)
        else:
            ret_msg = boss_aid.generate_init_message(message, n_results=n_results)
        return ret_msg if ret_msg else message

    boss_aid.human_input_mode = "NEVER"  # Disable human input for boss_aid since it only retrieves content.

    llm_config = {
        "functions": [
            {
                "name": "retrieve_content",
                "description": "retrieve content for code generation and question answering.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Refined message which keeps the original meaning and can be used to retrieve content for code generation and question answering.",
                        }
                    },
                    "required": ["message"],
                },
            },
        ],
        "config_list": config_list,
        "timeout": 60,
        "cache_seed": 42,
    }

    for agent in [coder, pm, reviewer]:
        # update llm_config for assistant agents.
        agent.llm_config.update(llm_config)

    for agent in [boss, coder, pm, reviewer]:
        # register functions for all agents.
        agent.register_function(
            function_map={
                "retrieve_content": retrieve_content,
            }
        )

    groupchat = autogen.GroupChat(
        agents=[boss, coder, pm, reviewer],
        messages=[],
        max_round=12,
        speaker_selection_method="random",
        allow_repeat_speaker=False,
    )

    manager_llm_config = llm_config.copy()
    manager_llm_config.pop("functions")
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=manager_llm_config)

    # Start chatting with the boss as this is the user proxy agent.
    boss.initiate_chat(
        manager,
        message=PROBLEM+" Must retrieve content using function calling.",
    )

print("FUNCTION CALLING RAG CHAT")
function_calling_rag_chat()
