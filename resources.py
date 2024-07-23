import sqlite3
from datetime import datetime
from playwright.sync_api import sync_playwright
import requests
from bs4 import BeautifulSoup

def initialize_db():
    conn = sqlite3.connect('resources.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS resources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            url TEXT NOT NULL UNIQUE,
            summary TEXT,
            screenshot BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def fetch_webpage_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # title = soup.find('title').get_text(strip=True)  
    text = soup.get_text(separator=' ', strip=True)  
    return text

def summarize_with_chatgpt(text, api_key):
    api_url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are an AI trained to summarize texts."},
            {"role": "user", "content": text}
        ]
    }
    response = requests.post(api_url, headers=headers, json=data)
    if response.status_code != 200:
        print("Failed to get a valid response from OpenAI API.")
        print("Status Code:", response.status_code)
        print("Response Content:", response.text)
        return "Failed to generate summary"

    try:
        summary = response.json()['choices'][0]['message']['content'].strip()
        return summary
    except KeyError:
        print("Unexpected JSON structure in the response.")
        print("JSON Response:", response.json())
        return "Error in summarizing content"

# def take_screenshot(url):
#     with sync_playwright() as playwright:
#         browser = playwright.chromium.launch(headless=True)
#         page = browser.new_page()
#         page.goto(url)
#         screenshot = page.screenshot(full_page=False)
#         browser.close()
#         return screenshot

def take_screenshot(url):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        title = page.title()
        screenshot = page.screenshot(full_page=False)
        browser.close()
        return title, screenshot


def save_to_database(title, url, summary, screenshot):
    conn = sqlite3.connect('resources.db')
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO resources (title, url, summary, screenshot)
        VALUES (?, ?, ?, ?)
    """, (title, url, summary, sqlite3.Binary(screenshot)))
    conn.commit()
    conn.close()

def scrape_and_store(url, api_key):
    conn = sqlite3.connect('resources.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM resources WHERE url = ?", (url,))
    exists = cursor.fetchone()
    if exists:
        print(f"URL already exists in the database: {url}")
        conn.close()
        return

    content = fetch_webpage_content(url)
    summary = summarize_with_chatgpt(content, api_key)
    title, screenshot = take_screenshot(url)
    save_to_database(title, url, summary, screenshot)
    conn.close()

def get_resources():
    conn = sqlite3.connect('resources.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, url, summary, created_at FROM resources")
    rows = cursor.fetchall()
    conn.close()
    return rows