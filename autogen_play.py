
"""
This module contains code for initializing an AssistantAgent and a UserProxyAgent, 
and performing a chat interaction between them.
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
    llm_config={
        "timeout": 3600,  # timeout for each request to OpenAI API
        "max_retries": 10,  # maximum number of retries for each request to OpenAI API
        "config_list": config_list,  # a list of OpenAI API configurations
        "max_tokens": 4096,  # maximum number of tokens for each response
        "temperature": 0  # temperature for sampling
    },
    is_termination_msg=lambda x: True if "TERMINATE" in x.get("content") else False
)

# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=20,
    is_termination_msg=lambda x: True if "TERMINATE" in x.get("content") else False,
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False
    },
    system_message=""
)

# the assistant receives a message from the user_proxy, which contains the task description
user_proxy.initiate_chat(
    assistant,
    message = """
I need assistance optimizing the damage output of my Dungeons & Dragons 5th Edition (D&D 5e) character. The character is an 11th-level Eldritch Knight equipped with a hand crossbow that has a +1 bonus to attack and regular hand crossbow damage. The character possesses the Crossbow Expert and Sharpshooter feats, along with the Archery combat style. Additionally, the character has a Dexterity score of 20.

To maximize damage per round, I would like to know for which enemy AC it is more efficient to use the Sharpshooter feat's ability to subtract 5 from the attack bonus and add 10 to damage. 

To achieve this, calculate the hit chance based on attack and enemy armor class by using the function below. Next multiply the result with the average damage to determine the average damage per attack. Do this both with and without utilizing the -5 attack/+10 damage ability. Present the results in a table detailing the average damage per attack for each enemy AC between 18 and 23.

def calculate_hit_chance(attack_bonus, enemy_ac):
    # Calculate hit chance
    roll_to_hit = max(1,enemy_ac - attack_bonus)
    hit_chance = 1-((1/20) * (roll_to_hit-1))
    return hit_chance
"""
)