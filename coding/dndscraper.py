"""
This module provides functions for scraping and searching Dungeons and Dragons 5th edition content from the dnd5e.wikidot.com website.
"""
import os
import json
import hashlib
import requests
from bs4 import BeautifulSoup
from googlesearch import search


def sanitize_html(html):
    """
    Remove streams, images, video, and other non-human-readable content from HTML.

    Args:
        html (str): The HTML content.

    Returns:
        sanitized_text (str): The sanitized text content.
    """
    soup = BeautifulSoup(html, 'html.parser')

    # Remove script, style, object, embed, applet, audio, video, iframe, img tags
    for tag in soup(["script", "style", "object", "embed", "applet", "audio", "video", "iframe", "img"]):
        tag.decompose()

    # Get text content
    text_content = soup.get_text()

    return text_content.strip()


def search_dnd5e_subpages(query, num_results=5):
    """
    Searches subpages of http://dnd5e.wikidot.com for the given query and returns a JSON list of objects
    with the title of the page as the key and sanitized text content as the value.

    Args:
        query (str): The search query.
        num_results (int, optional): The number of Google search results to access. Defaults to 5.

    Returns:
        results (list): A list of dictionaries with 'title' and 'content' keys.

    Example:
        >>> results = search_dnd5e_subpages("Dungeons and Dragons 5th edition rules")
        >>> print(results)
    """

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

# Example usage:
#my_query = "Eldritch Knight"
#my_num_results = 5
#my_results = search_dnd5e_subpages(my_query, my_num_results)
#print(my_results)
