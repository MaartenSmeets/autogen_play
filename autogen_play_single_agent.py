
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
    is_termination_msg=lambda x: True if "TERMINATE" in x.get("content") else False,
    system_message="""You use your language skills and write python/shell code to solve tasks provided to you. When you need to collect info, suggest code to obtain the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. Suggest code to finish the task smartly. Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
Reply "TERMINATE" when the question asked has been answered by the output of your code.
"""
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
    }
)

# the assistant receives a message from the user_proxy, which contains the task description
user_proxy.initiate_chat(
    assistant,
    message = """
I need assistance optimizing the damage output of my Dungeons & Dragons 5th Edition character. The character is an 11th-level Eldritch Knight equipped with a hand crossbow. The character possesses the Crossbow Expert and Sharpshooter feats, along with the Archery combat style. Additionally, the character has a Dexterity score of 20.

To maximize damage per round, I would like to know for which enemy AC it is more efficient to use the Sharpshooter feat's ability to subtract 5 from the attack bonus and add 10 to damage. Include information from Dungeons Masters Guide, Players Handbook, Xanathar's Guide to Everything (XGE), Tasha's Cauldron of Everything (TCE) and exclude any homebrew.
"""
)