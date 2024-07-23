import requests
import re

# Function to strip HTML tags from text
def strip_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

# apiKey = "D5TvzaGcYfx4GOxOn834UD9QxCAyhEAH"
# subscriptionKey = "f3f0023662b94a9cbfefa2b60472122e"

def newsSearch(api_key, searchTerm):
    search_url = "https://api.bing.microsoft.com/v7.0/news/search"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {
        "q": f"{searchTerm} stock market finance",
        "textDecorations": True,
        "textFormat": "HTML",
        "mkt": "en-US",
        "count": 10  # Adjusted for more relevant results
    }

    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()  # Raises an HTTPError for bad requests
    results = response.json()
    stories = results['value']

    resultList = []
    for story in stories:
        image_url = story.get('image', {}).get('thumbnail', {}).get('contentUrl', 'default-image.png')
        
        resultList.append({
            'name': story['name'],
            'url': story['url'],
            'image': image_url,
            'description': strip_html_tags(story['description']),
            'provider': story['provider'][0]['name'],
            'date': story['datePublished']
        })

    return resultList
