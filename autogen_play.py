
"""
This module contains code for initializing an AssistantAgent and a UserProxyAgent, and performing a chat interaction between them.
"""

import autogen

config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["local"]
    },
)

# create an AssistantAgent named "assistant"
assistant = autogen.AssistantAgent(
    name="assistant",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: True if "TERMINATE" in x.get(
        "content") else False,
    llm_config={
        "timeout": 900,  # timeout for each request to OpenAI API
        "max_retries": 10,  # maximum number of retries for each request to OpenAI API
        "config_list": config_list,  # a list of OpenAI API configurations
        "max_tokens": 4096,  # maximum number of tokens for each response
        "temperature": 0,  # temperature for sampling
    },
    system_message="You are a helpful assistant that can use available functions when needed to solve problems. At each point, do your best to determine if the user's request has been addressed. IF THE REQUEST HAS NOT BEEN ADDRESSED, RESPOND WITH CODE TO ADDRESS IT. IF A FAILURE OCCURRED (e.g., due to a missing library) AND SOME ADDITIONAL CODE WAS WRITTEN (e.g. code to install the library), ENSURE THAT THE ORIGINAL CODE TO ADDRESS THE TASK STILL GETS EXECUTED. If the request HAS been addressed, respond with a summary of the result. The summary must be written as a coherent helpful response to the user request e.g. 'Sure, here is result to your request ' or 'The tallest mountain in Africa is ..' etc. The summary MUST end with the word TERMINATE. If the  user request is pleasantry or greeting, you should respond with a pleasantry or greeting and TERMINATE. The task you will help to solve is that you will answer questions related to DnD. Write Python code to fetch relevant references using the dndscraper module search_dnd5e_subpages(query) function. Evaluate the results before you use it in answering questions. Include Dungeons Masters Guide, Players Handbook, Xanathar's Guide to Everything (XGE), Tasha's Cauldron of Everything (TCE) and exclude any homebrew. If a table needs to be created, write Python code to do this and document your code."
)

# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: True if "TERMINATE" in x.get(
        "content") else False,
    code_execution_config={
        "work_dir": "coding",
        "use_docker": True,
    }
)

# the assistant receives a message from the user_proxy, which contains the task description
user_proxy.initiate_chat(
    assistant,
    message = """
    As an expert in game system calculations and statistics, I need assistance optimizing the damage output of my Dungeons & Dragons 5th Edition (D&D 5e) character. The character is an 11th-level Eldritch Knight with a proficiency bonus of +4. Equipped with a hand crossbow that has a +1 bonus to attack, the character possesses the Crossbow Expert and Sharpshooter feats, along with the Archery combat style. Additionally, the character boasts a Dexterity score of 20.

    To maximize damage per round, I would like to create a table for enemy armor classes ranging from 18 to 23. The objective is to determine whether using the Sharpshooter feat's ability to subtract 5 from the attack bonus and add 10 to damage would optimize the damage output. The calculations involve factoring in critical damage and critical misses in the average damage calculation.

    To achieve this, I'll calculate the hit chance and then multiply it by the average damage to determine the average damage per attack. The resulting table will include hit chance values both with and without utilizing the -5 attack/+10 damage ability. Additionally, it will provide average damage per attack on a hit, both with and without the ability. Finally, the table will showcase the overall average damage per attack by multiplying the average damage on hit with the hit chance for both scenarios, with and without using the -5 attack/+10 damage ability.
    """
)

