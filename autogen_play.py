
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
        "use_docker": True,
    },
    system_message=""
)

# the assistant receives a message from the user_proxy, which contains the task description
user_proxy.initiate_chat(
    assistant,
    message = """
    I need assistance optimizing the damage output of my Dungeons & Dragons 5th Edition (D&D 5e) character. The character is an 11th-level Eldritch Knight. Equipped with a hand crossbow that has a +1 bonus to attack and regular hand crossbow damage, the character possesses the Crossbow Expert and Sharpshooter feats, along with the Archery combat style. Additionally, the character boasts a Dexterity score of 20.

    To maximize damage per round, I would like to know for which enemy AC between 18 and 23 it is more efficient to use the Sharpshooter feat's ability to subtract 5 from the attack bonus and add 10 to damage. The calculations involve factoring in critical damage and critical misses in the average damage calculation.

    To achieve this, calculate the hit chance based on attack and enemy armor class and then multiply it by the average damage to determine the average damage per attack. Do this both with and without utilizing the -5 attack/+10 damage ability and compare the results to determine whether using the ability maximizes damage per attack.
    
    You can use the following code to help you complete your task and fetch additional specific information relevant for answering my request such as information about the Eldritch Knight or information about a hand crossbow. Before using information validate its source. Include Dungeons Masters Guide, Players Handbook, Xanathar's Guide to Everything (XGE), Tasha's Cauldron of Everything (TCE) and exclude any homebrew.

import os
import json
import hashlib
import requests
from bs4 import BeautifulSoup
from googlesearch import search

def sanitize_html(html):
    soup = BeautifulSoup(html, 'html.parser')

    # Remove script, style, object, embed, applet, audio, video, iframe, img tags
    for tag in soup(["script", "style", "object", "embed", "applet", "audio", "video", "iframe", "img"]):
        tag.decompose()

    # Get text content
    text_content = soup.get_text()

    return text_content.strip()

def search_dnd5e_subpages(query, num_results=1):
    key = hashlib.md5(("search_dnd5e_subpages(" + str(num_results) + ")" + query).encode("utf-8"))
    cache_dir = ".cache"
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)

    fname = os.path.join(cache_dir, key.hexdigest() + ".cache")

    # Cache hit
    if os.path.isfile(fname):
        fh = open(fname, "r", encoding="utf-8")
        data = json.loads(fh.read())
        fh.close()
        return data

    results = []

    # Google search for subpages of http://dnd5e.wikidot.com
    site_search_query = f"{query} site:dnd5e.wikidot.com"
    for url in search(site_search_query, num_results=num_results):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            sanitized_content = sanitize_html(response.text)
            title = BeautifulSoup(response.text, 'html.parser').title.text if BeautifulSoup(response.text, 'html.parser').title else 'No Title'

            results.append({
                'title': title.strip(),
                'content': sanitized_content,
            })

        except requests.exceptions.RequestException as e:
            print(f"Error accessing {url}: {e}")

    # Save to cache
    fh = open(fname, "w", encoding="utf-8")
    fh.write(json.dumps(results))
    fh.close()
    for file in os.listdir(cache_dir):
        file_path = os.path.join(cache_dir, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except OSError as e:
            print(f"Error deleting {file_path}: {e}")
    return results
    """
)
