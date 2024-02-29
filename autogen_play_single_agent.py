
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
        "temperature": 0.5,  # temperature for sampling
        "cache_seed": None
    },
    is_termination_msg=lambda x: x.get(
        "content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    system_message="""You are a helpful AI assistant.
Solve tasks using your coding and language skills.
In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.
    1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
    2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
If you want the user to save the code in a file before executing it, start with three times single quote. Next put # filename: <filename> inside the code block as the first line. End the code block with three times single quote. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user. Do not simulate code output but only write code blocks conforming to previously mentioned standards.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
Only reply "TERMINATE" in the end when everything is done and the complete task/plan has been completed. Do not show appreciation in your responses. Do not comment on whether the task is ethic or legal. Say only what is necessary.
"""
)

# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=40,
    is_termination_msg=lambda x: x.get(
        "content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False
    }
)

# the assistant receives a message from the user_proxy, which contains the task description
user_proxy.initiate_chat(
    assistant,
    message="""
- Begin by fetching the website https://5thsrd.org/rules/leveling_up/ and saving the HTML locally to a file. Confirm the HTML file has been created and contains data and only then continu with the next step. If the file is missing or if it is empty, investigate previous code and execution results to identify and rectify any issues.
- Analyse the first table in the locally saved HTML file (without assuming a table class) by printing it to the console. Your goal of analysing the table is to create a parser which extracts the data that contains the XP requirements for each character level. Take into account the XP in the table uses a comma to separate thousands.
- Use the previously determined data and save it to a CSV file. Confirm the CSV file has been created and contains data and only then continu with the next step. If the file is missing or if it is empty, investigate previous code and execution results to identify and rectify any issues.
- Utilize the contents of the CSV file to create a line chart using the matplotlib library. The x-axis should represent character levels, while the y-axis should indicate the required XP for each level. Export the graph as PNG. Write your code to not require user interaction. Confirm the PNG file has been created and only then continu with the next step. If the file is missing, investigate previous code and execution results to identify and rectify any issues.
- Consider the task complete only when the PNG file creation is confirmed. If this is the case, reply "TERMINATE” to signal the end of the task.
"""
)