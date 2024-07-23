import os
import re
import io
import json
import requests
import sqlite3
import certifi
import secrets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from PIL import Image
from io import BytesIO
from urllib.request import urlopen
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify, send_file, Response
from flask_behind_proxy import FlaskBehindProxy
from textblob import TextBlob
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from playwright.sync_api import sync_playwright
from resources import initialize_db, scrape_and_store, get_resources
import openai
from forms import userPrompt
import PyQt5
from news import newsSearch

app = Flask(__name__)

# Initialize the database
initialize_db()

# URLs for the resources webpage
urls = [
    "https://www.synovus.com/personal/resource-center/investing/investing-101-understanding-the-stock-market/",
    "https://www.bankrate.com/investing/ultimate-guide-virtual-trading-stock-market-simulator/",
    "https://www.bankrate.com/investing/stock-market-basics-for-beginners/",
    "https://www.nerdwallet.com/article/investing/stock-market-basics-everything-beginner-investors-know",
    "https://www.bankrate.com/investing/how-to-read-stock-charts/"
]

api_key = os.getenv('OPENAI_KEY')

for url in urls:
    scrape_and_store(url, api_key)

apiKey = "D5TvzaGcYfx4GOxOn834UD9QxCAyhEAH"
subscriptionKey = "f3f0023662b94a9cbfefa2b60472122e"

# Function to get price change
def getPriceChange(stockSymbol):
    companyPriceURL = f"https://financialmodelingprep.com/api/v3/stock-price-change/{stockSymbol}?apikey={apiKey}"
    response = urlopen(companyPriceURL, cafile=certifi.where())
    data = response.read().decode("utf-8")
    jsonOfCompanies = json.loads(data)
    return jsonOfCompanies

# Function to get all company names
def getAllCompanies():
    companySearchUrl = f'https://financialmodelingprep.com/api/v3/stock/list?apikey={apiKey}'
    response = urlopen(companySearchUrl, cafile=certifi.where())
    data = response.read().decode("utf-8")
    jsonOfCompanies = json.loads(data)
    return jsonOfCompanies

# Function specifically made to make an API call
def stockApiCall(nameOfCompany, option):
    if option == 1:
        generalSearchUrl = f'https://financialmodelingprep.com/api/v3/search?query={nameOfCompany}&limit=3&&apikey={apiKey}'
        response = urlopen(generalSearchUrl, cafile=certifi.where())
        data = response.read().decode("utf-8")
        jsonOfCompanies = json.loads(data)
        return jsonOfCompanies
    elif option == 2:
        companyProfileUrl = f'https://financialmodelingprep.com/api/v3/profile/{nameOfCompany}?apikey={apiKey}'
        response = urlopen(companyProfileUrl, cafile=certifi.where())
        data = response.read().decode("utf-8")
        jsonOfCompanies = json.loads(data)
        return jsonOfCompanies
    elif option == 3:
        today = datetime.now()
        one_month_ago = today - relativedelta(months=1)
        today_str = today.strftime('%Y-%m-%d')
        one_month_ago_str = one_month_ago.strftime('%Y-%m-%d')
        companyHistoryPriceUrl = f'https://financialmodelingprep.com/api/v3/historical-chart/1hour/{nameOfCompany}?from={one_month_ago_str}&to={today_str}&apikey={apiKey}'
        response = urlopen(companyHistoryPriceUrl, cafile=certifi.where())
        data = response.read().decode("utf-8")
        jsonOfCompanies = json.loads(data)
        return jsonOfCompanies

# Function to make line graph from the time and quotes of a company
def graphData(independant, dependant, symbolName, prices, companyName):
    matplotlib.use('qtagg')
    plt.clf()
    plt.rc('font', size=8)
    x = np.array(independant)
    y = np.array(dependant)
    today = datetime.now()
    one_month_ago = today - timedelta(days=30)
    currPriceChange = float(prices['1M'])
    line_color = 'green' if currPriceChange > 0 else 'red'
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color=line_color), name=f"{companyName}'s Prices"))
    fig.update_xaxes(
        title_text="Dates",
        tickformat="%Y-%m-%d",
        tickangle=45,
        range=[one_month_ago, today],
        tickmode='linear',
        dtick=86400000.0 * 2  # Tick every other day
    )
    fig.update_yaxes(title_text="Prices")
    fig.update_layout(title=f"{companyName}'s Prices", margin=dict(l=0, r=0, t=30, b=0))
    graph_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    return f'<div style="width:50%;">{graph_html}</div>'

# Function to find stock information on company
def getCompanyInfo(nameOfCompany):
    jsonOfCompanies = stockApiCall(nameOfCompany, 1)
    listOfCompanies = {companyDict['name']: companyDict['symbol'] for companyDict in jsonOfCompanies}
    print(listOfCompanies)
    nameOfCompany = input("Pick one from the names provided \n")
    if nameOfCompany not in listOfCompanies:
        print("bruh lock in")
        return
    else:
        companySymbol = listOfCompanies[nameOfCompany]
        companyProfile = stockApiCall(companySymbol, 2)
        currentPrice = companyProfile[0]['price']
        print(f'the price of {companySymbol} is {currentPrice}')
        quoteJson = stockApiCall(companySymbol, 3)
        df = pd.json_normalize(quoteJson)
        graphData(df['date'].head(5), df['open'].head(5), nameOfCompany)

# Function to return a graph to a page based on the word that was searched
def name_to_graph(companySymbol):
    quoteJson = stockApiCall(companySymbol, 3)
    changeInPrice = getPriceChange(companySymbol)
    profileJson = stockApiCall(companySymbol, 1)
    profileDF = pd.json_normalize(profileJson)
    prices = pd.json_normalize(changeInPrice)
    df = pd.json_normalize(quoteJson)
    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'].dt.date
    companyName = profileDF['name'][0]
    res = graphData(df['date'], df['open'], companySymbol, prices, companyName)
    return res

# Setting up the database
conn = sqlite3.connect('personal-portfolio.db', check_same_thread=False)
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS portfolio (
    id Integer PRIMARY KEY,
    name TEXT NOT NULL,
    price Integer,
    changeInPrice Integer
    )
''')

# Adds the stock onto the personal portfolio db
def addDatabase(name, price, changeInPrice):
    c.execute('''
    INSERT INTO portfolio(name, price, changeInPrice)
    VALUES (?, ?, ?)
    ''', (name, price, changeInPrice))

# Setting up chatBot
my_api_key = os.getenv('OPENAI_KEY')
openai.api_key = my_api_key

app = Flask(__name__)
key = secrets.token_hex(16)
proxied = FlaskBehindProxy(app)
app.config['SECRET_KEY'] = key

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/portfolio')
def portfolio():
    return render_template('portfolio.html')

@app.route('/market', methods=['GET', 'POST'])
def market():
    form = userPrompt()
    if request.method == 'POST':
        user_requested_company = form.getName()
        if len(user_requested_company) > 0:
            plot_html = name_to_graph(user_requested_company)
            return render_template('market.html', form=form, word=user_requested_company, plot_html=plot_html)
    return render_template('market.html', form=form)

def get_resources():
    conn = sqlite3.connect('resources.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, url, summary FROM resources")
    rows = cursor.fetchall()
    conn.close()
    return rows

@app.route('/resources', methods=['GET'])
def resources():
    url_summaries = get_resources()
    return render_template('resources.html', url_summaries=url_summaries)

@app.route('/screenshot/<int:resource_id>')
def screenshot(resource_id):
    conn = sqlite3.connect('resources.db')
    cursor = conn.cursor()
    cursor.execute("SELECT screenshot FROM resources WHERE id = ?", (resource_id,))
    row = cursor.fetchone()
    conn.close()
    if row and row[0]:
        return send_file(BytesIO(row[0]), mimetype='image/jpeg')
    else:
        return "No image available", 404

@app.route('/news')
def news():
    return render_template('news.html')

@app.route('/get_news', methods=['GET'])
def get_news():
    company = request.args.get('company')
    news_data = newsSearch(subscriptionKey, company)
    if news_data:
        articles = [{"name": article['name'], "url": article['url'], "description": article['description']} for article in news_data]
        return jsonify(articles)
    else:
        return jsonify({"error": "Failed to fetch news"}), 500

@app.route('/chatbot', methods=['POST'])
def chat_bot():
    user_message = request.form['message']
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_message}
            ]
        )
        ai_response = response['choices'][0]['message']['content']
        return jsonify({'response': ai_response})
    except Exception as e:
        return jsonify({'response': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
