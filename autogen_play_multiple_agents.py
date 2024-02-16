
"""
This module contains code for initializing an AssistantAgent and a UserProxyAgent, 
and performing a chat interaction between them.
"""

import autogen
import textwrap
import logging

#autogen.logging.basicConfig(level=autogen.logging.DEBUG)
#http.client.HTTPConnection.debuglevel = 1

config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["local"]
    }
)

llm_config = {
    "timeout": 7200,  # timeout for each request to OpenAI API
    "max_retries": 10,  # maximum number of retries for each request to OpenAI API
    "config_list": config_list,  # a list of OpenAI API configurations
    "max_tokens": 4096,  # maximum number of tokens for each response
    "temperature": 0.5,  # temperature for sampling
    "cache_seed": None
}

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get(
        "content", "").rstrip().endswith("TERMINATE"),
    llm_config=llm_config,
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    }
)

coder = autogen.AssistantAgent(
    name="Coder",  # the default assistant agent is capable of solving problems with code
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get(
        "content", "").rstrip().endswith("TERMINATE"),
    llm_config=llm_config,
    system_message="""You are a helpful AI assistant.
Solve tasks using your coding and language skills.
In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.
    1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
    2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
Reply "TERMINATE" in the end when everything is done.
    """
)



critic = autogen.AssistantAgent(
    name="Critic",
    system_message="""Critic. You are a helpful assistant highly skilled in evaluating the quality of a given visualization code by providing a score from 1 (bad) - 10 (good) while providing clear rationale. YOU MUST CONSIDER VISUALIZATION BEST PRACTICES for each evaluation. Specifically, you can carefully evaluate the code across the following dimensions
- bugs (bugs):  are there bugs, logic errors, syntax error or typos? Are there any reasons why the code may fail to compile? How should it be fixed? If ANY bug exists, the bug score MUST be less than 5.
- Data transformation (transformation): Is the data transformed appropriately for the visualization type? E.g., is the dataset appropriated filtered, aggregated, or grouped  if needed? If a date field is used, is the date field first converted to a date object etc?
- Goal compliance (compliance): how well the code meets the specified visualization goals?
- Visualization type (type): CONSIDERING BEST PRACTICES, is the visualization type appropriate for the data and intent? Is there a visualization type that would be more effective in conveying insights? If a different visualization type is more appropriate, the score MUST BE LESS THAN 5.
- Data encoding (encoding): Is the data encoded appropriately for the visualization type?
- aesthetics (aesthetics): Are the aesthetics of the visualization appropriate for the visualization type and the data?

YOU MUST PROVIDE A SCORE for each of the above dimensions.
{bugs: 0, transformation: 0, compliance: 0, type: 0, encoding: 0, aesthetics: 0}
Do not suggest code. Do not evaluate code but only execution results.
Finally, based on the critique above, suggest a concrete list of actions that the coder should take to improve the code.
""",
    llm_config=llm_config,
    is_termination_msg=lambda x: x.get(
        "content", "").rstrip().endswith("TERMINATE"),
)

groupchat = autogen.GroupChat(
    agents=[user_proxy, coder, critic], messages=[], max_round=20)

planner = autogen.GroupChatManager(
    name="Planner",  # the default assistant agent is capable of solving problems with code
    human_input_mode="NEVER",
    groupchat=groupchat,
    llm_config=llm_config,
    is_termination_msg=lambda x: x.get(
        "content", "").rstrip().endswith("TERMINATE"),
    system_message="""Your role is to coordinate AI assistants to achieve a task and determine if it is completed. 
    Determine who should be the next assistant. The available assistants are a 'Coder' who can write code, 'Critic' 
    who can evaluate code output and provide feedback. 'user_proxy' who can execute code. Determine whether the request has 
    been fulfilled or the question has been answered and there are no open issues or suggestions for improvements by the Critic. 
    Do not simulate interaction and do not answer questions which should be answered by the Critic, Coder or user_proxy Assistant. 
    Do not provide a conversation or dialog with the other assistants. Reply with TERMINATE when done.",
"""
    )

user_proxy.initiate_chat(
    planner,
    message="parse data on xp per level for dnd 5th edition from the following website https://roll20.net/compendium/dnd5e/Character%20Advancement#content and visualize it in a bar chart. Do not show appreciation in your responses, say only what is necessary. "
    "if 'Thank you' or 'You're welcome' are said in the conversation, then say TERMINATE "
    "to indicate the conversation is finished and this is your last message.",
)
# type exit to terminate the chat
