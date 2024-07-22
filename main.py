
from flask import Flask, render_template, request, jsonify
import requests
import re

app = Flask(__name__)

apiKey = "D5TvzaGcYfx4GOxOn834UD9QxCAyhEAH"
subscriptionKey = "f3f0023662b94a9cbfefa2b60472122e"


def newsSearch(searchTerm):
    search_url = "https://api.bing.microsoft.com/v7.0/news/search"
    headers = {"Ocp-Apim-Subscription-Key": subscriptionKey}
    params = {
        "q": f"{searchTerm} stock market finance",
        "textDecorations": True,
        "textFormat": "HTML",
        "mkt": "en-US",
        "count": 20  # Increase count to get more results and filter later
    }

    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    results = response.json()
    stories = results['value']

    def strip_html_tags(text):
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)

    resultList = []
    for story in stories:
        image_url = story.get('image', {}).get('thumbnail', {}).get('contentUrl', '')
        # Check for a larger image URL if available in the API response
        if 'contentUrl' in story.get('image', {}):
            image_url = story['image']['contentUrl']
        resultList.append({
            'name': story['name'],
            'url': story['url'],
            'image': image_url,
            'description': strip_html_tags(story['description']),
            'provider': story['provider'][0]['name'],
            'date': story['datePublished']
        })

    return resultList


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')
    
@app.route('/portfolio')
def portfolio():
    return render_template('portfolio.html')

@app.route('/market')
def market():
    return render_template('market.html')

@app.route('/resources')
def learn():
    return render_template("resources.html")

@app.route('/news')
def news():
    return render_template('news.html')

@app.route('/get_news', methods=['GET'])
def get_news():
    company = request.args.get('company')
    news_data = newsSearch(company)
    articles = [
        {"name": article['name'], "url": article['url'], "image": article['image'], "description": article['description']}
        for article in news_data
    ]
    return jsonify(articles)

if __name__ == '__main__':
    app.run(debug=True)
