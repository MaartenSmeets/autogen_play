
"""
This module contains code for initializing an AssistantAgent and a UserProxyAgent, 
and performing a chat interaction between them.
"""

import autogen

config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["local"]
    }
)

llm_config={
        "timeout": 3600,  # timeout for each request to OpenAI API
        "max_retries": 10,  # maximum number of retries for each request to OpenAI API
        "config_list": config_list,  # a list of OpenAI API configurations
        "max_tokens": 4096,  # maximum number of tokens for each response
        "temperature": 0  # temperature for sampling
    }

engineer = autogen.AssistantAgent(
    name="Engineer",
    human_input_mode="NEVER",    
    llm_config=llm_config,
    system_message="""Engineer. You use your language skills and write python/shell code to solve tasks provided to you. When you suggest code, the Executor executes it. The Tester tests the code and provides feedback. Use this feedback to correct the code. When you need to collect info, suggest code to obtain the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. Suggest code to finish the task smartly. Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try. If the result you obtain from the Executor who executes your suggested code block and returns you the result, indicates there is an error, fix the error and output the code again. 
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible. Reply "TERMINATE" in the end when the initial question has been answered by the output of the code you suggested.
"""
)

tester = autogen.AssistantAgent(
    name="Tester",
    human_input_mode="NEVER",    
    llm_config=llm_config,
    system_message="""Tester. You use your language skills and expertise in functional requirements to either approve the code output as returned by the Executor or reject the code with an elaborate reason detailing why the code does not conform to requirements. Return your response to the Engineer. When the code you receive from the Executor returns an error, return the error to the Engineer. You do not write code yourself but provide feedback to the Engineer on the code output created by the Executor. 
    Use the following requirements and check the numbers which you expect with the results provided by the Executor. If different than reject the output and provide an elaborate reason for rejection.
    
    The hit chance calculation based on enemy AC and attack bonus should give the following results
    - the hit chance should be between 0 and 1
    - when the enemy AC is 20 and the attack bonus is 19, the hit chance is 1
    - when the enemy AC is higher than the attack bonus + 20, the hit chance is 0. for example, when the attack bonus is 2 and the enemy ac is 23, the hit chance is 0
    - when the enemy AC is 15 and the attack bonus is 10, the hit chance is 0.75
    - when the enemy AC is 20 and the attack bonus is 10, the hit chance is 0.50

    The average damage per attack on hit should be calculated as follows
    - the average damage without the sharpshooter feat on hit with a hand crossbow and a dexterity modifyer of 5 is (1d6 -> 1 + 6) / 23.5+5=8.5
    - the average damage with the sharpshooter feat on hit with a hand crossbow and a dexterity modifyer of 5 is 3.5+5+10=18.5
"""
)

executor = autogen.AssistantAgent(
    name="Executor",
    llm_config=llm_config,
    system_message="Executor. Execute the code written by the Engineer and report the result to the Tester",
    human_input_mode="NEVER",
    code_execution_config={
        "last_n_messages": 3,
        "work_dir": "coding",
        "use_docker": False
    }  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)

user_proxy = autogen.UserProxyAgent(
    name="User",
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=20,
    is_termination_msg=lambda x: True if "TERMINATE" in x.get("content") else False,
    code_execution_config=False
)


groupchat = autogen.GroupChat(
    agents=[engineer, executor, tester], messages=[], max_round=50, speaker_selection_method="round_robin"
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

user_proxy.initiate_chat(
    manager,
    message="""
I need assistance optimizing the damage output of my Dungeons & Dragons 5th Edition character. The character is an 11th-level Eldritch Knight equipped with a hand crossbow (3.5 on average base damage). The character possesses the Crossbow Expert and Sharpshooter feats, along with the Archery combat style. Additionally, the character has a Dexterity score of 20.

To maximize damage per round, I would like to know for which enemy AC it is more efficient to use the Sharpshooter feat's ability to subtract 5 from the attack bonus and add 10 to damage. Create a table with average damage per attack and multiply this with the hit chance for enemy AC 15 to 23.
"""
)
