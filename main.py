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
import matplotlib
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
# from openai import OpenAI
from forms import userPrompt
import PyQt5
from news import newsSearch
import openai
from add import addPortfolio

app = Flask(__name__)

#validators
stock_data = ["", "", ""]

# Initialize the database
initialize_db()
# urls for the resources webpage
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

apiKey = "P8Fzcp6DjIGHvtIbM52L3YGDvJaD3HkZ"
subscriptionKey = "f3f0023662b94a9cbfefa2b60472122e"

# function to get price change
def getPriceChange(stockSymbol):

    companyPriceURL = (f"https://financialmodelingprep.com/api/v3/stock-price-change/{stockSymbol}?apikey={apiKey}")

    response = urlopen(companyPriceURL, cafile=certifi.where())

    data = response.read().decode("utf-8")
    
    jsonOfCompanies = json.loads(data)
    
    return jsonOfCompanies

# function to get all company names
def getAllCompanies():
    companySearchUrl = (f'https://financialmodelingprep.com/api/v3/stock/list?apikey={apiKey}')
        
    response = urlopen(companySearchUrl, cafile=certifi.where())
    
    data = response.read().decode("utf-8")
    
    jsonOfCompanies = json.loads(data)
    
    return jsonOfCompanies

# function to search for news on a company 
# def newsSearch(searchTerm):
#     search_url = "https://api.bing.microsoft.com/v7.0/news/search"

#     headers = {"Ocp-Apim-Subscription-Key" : subcriptionKey}
#     params  = {"q": searchTerm, "textDecorations": True, "textFormat": "HTML"}

#     response = requests.get(search_url, headers=headers, params=params)
#     response.raise_for_status()
#     results = response.json()

#     # this will get you the first url
#     # print(results['value'][0]['url'])

#     # a list of the news articles from the searched word
#     stories = results['value']

#     resultList = []

#     for story in stories:

#         resultList.append( { 'name' : story['name'], 'url' : story['url'], 'image' : story['image']['thumbnail']['contentUrl'], 'description' : story['description'], 'provider' : story['provider'][0]['name'], 'data' : story['datePublished']} )
    
#     # print(resultList[0])

#     return resultList


    # search_results = json.dumps(response.json())

   
    # pprint.pp(search_results[0])

# # function to analyze the sentiment of the aritcle
# FUCK
# def sentimentAnalysis(url):

#     # USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15'

#     # config = Config()
#     # config.browser_user_agent = USER_AGENT
#     # config.request_timeout = 10

#     # article = Article(url, config=config)
#     article = newspaper.article(url)
#     article.download()
#     article.parse()
#     article.nlp()

#     text = article.summary
#     print(article.summary)
#     blob = TextBlob(text)

#     sentiment = blob.sentiment.polarity

#     print(sentiment)

# function specifically made to make an api call
def stockApiCall(nameOfCompany, option):
    if option == 1:
        # use api to return a list of company names that match the name 

        generalSearchUrl = (f'https://financialmodelingprep.com/api/v3/search?query={nameOfCompany}&limit=3&&apikey={apiKey}')
        
        # turn the request into json format

        response = urlopen(generalSearchUrl, cafile=certifi.where())
       
        data = response.read().decode("utf-8")
       
        jsonOfCompanies = json.loads(data)
        
        return jsonOfCompanies
    
    elif option == 2:
        # use api to return a list of company profiles that match the name 
       
        companyProfileUrl = (f'https://financialmodelingprep.com/api/v3/profile/{nameOfCompany}?apikey={apiKey}')
        
        # turn the request into json format

        response = urlopen(companyProfileUrl, cafile=certifi.where())
        
        data = response.read().decode("utf-8")
        
        jsonOfCompanies = json.loads(data)
        
        return jsonOfCompanies
    
    elif option == 3:

        # Get today's date
        today = datetime.now()

        # Calculate the date one month ago
        one_month_ago = today - relativedelta(months=1)

        # Format both dates as yyyy-mm-dd
        today_str = today.strftime('%Y-%m-%d')
        # print(today_str)
        one_month_ago_str = one_month_ago.strftime('%Y-%m-%d')
        # print(one_month_ago_str)



        companyHistoryPriceUrl = (f'https://financialmodelingprep.com/api/v3/historical-chart/1hour/{nameOfCompany}?from={one_month_ago_str}&to={today_str}&apikey={apiKey}')

        # turn the request into json format

        response = urlopen(companyHistoryPriceUrl, cafile=certifi.where())
        
        data = response.read().decode("utf-8")
        
        jsonOfCompanies = json.loads(data)
        
        return jsonOfCompanies
    


# function to make line graph from the time and quotes of a company
def graphData(independant, dependant, symbolName, prices, companyName):

    # print(f"PRICES ARE {prices}")
    matplotlib.use('qtagg')

    plt.clf()

    # fig = Figure()

    # fig.suptitle(f"{companyName}'s Quotas")

    # axis = fig.add_subplot(1, 1, 1)

    # fig, ax = plt.subplots(figsize=(10, 6))


    plt.rc('font', size=8)    # font size

    x = np.array(independant)

    y = np.array(dependant)

    today = datetime.now()
    print(f"TODAY IS {today}")
    one_month_ago = today - timedelta(days=30)

    # Calculate the slope
    # Use np.polyfit to fit a line (degree=1) to the price data
    # slope, intercept = np.polyfit(range(len(y)), y, 1)
    currPriceChange = float(prices['1M'])
    # Set the line color based on the slope
    line_color=''
    if currPriceChange > 0:
        line_color = 'green'
    else:
        line_color = 'red'

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color=line_color), name=f"{companyName}'s Prices"))
    
    # fig.update_xaxes(title_text="Dates", tickformat="%Y-%m-%d", dtick="86400000.0*2", tickangle=45)
    fig.update_xaxes(
        title_text="Dates",
        tickformat="%Y-%m-%d",
        tickangle=45,
        range=[one_month_ago, today],
        tickmode='linear',
        dtick=86400000.0 * 2  # Tick every other day
    )
    fig.update_yaxes(title_text="Prices")
    fig.update_layout(title=f"{companyName}'s Prices", margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor='rgba(235, 241, 254, 0.3)')

    # ax.plot(x, y, color=line_color)

    # ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))

    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # plt.xticks(rotation=45, ha='right')

    # today = datetime.now()
    # one_month_ago = today - timedelta(days=30)
    # ax.set_xlim(one_month_ago, today)

    # plt.plot(x,y)

    # plt.xlabel("Dates")  # add X-axis label
    # plt.ylabel("Prices")  # add Y-axis label
    # plt.title(f"{companyName}'s prices")  # add title
   

    # plt.show()

    # axis.set_xlabel("Dates")
    # axis.set_ylabel("Prices")

    # axis.plot(x, y)

    # return fig
    # plt.show()
    # plt.tight_layout()
    # plt.savefig('static/images/new_plot.png')

     # Define the date range for the last 30 days
    # today = datetime.now()
    # one_month_ago = today - timedelta(days=30)
    # fig.update_xaxes(range=[one_month_ago, today])
    graph_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    return f'<div style="width:50%;">{graph_html}</div>'

    
# function to find stock information on comapny
def getCompanyInfo(nameOfCompany):

    returnVal = ["", "", ""]
    try:  
        jsonOfCompanies = stockApiCall(nameOfCompany, 1)
    except:
        return [" "," "," "]

    jsonOfCompanies = stockApiCall(nameOfCompany, 1)

    # print(jsonOfCompanies[0]['name'])

    listOfCompanies = {}

    for companyDict in jsonOfCompanies:
       
        resultName = companyDict['name']
        if resultName not in listOfCompanies:
            listOfCompanies[resultName] = companyDict['symbol']

    # list of companies that match what the user wanted 
    print(listOfCompanies)

    if listOfCompanies is None:
        return [" "," "," "]

    # now we are going to have the user choose what they wanted out of the list
    

    if listOfCompanies is None:
        # edge case if the user picks something random

        print("bruh lock in")

        return
    else:

        print(listOfCompanies)
        if len(listOfCompanies) < 1:
             
             return ["","",""]
        
        first_key = list(listOfCompanies)[0]
        first_val = list(listOfCompanies.values())[0]
        print(first_key)
        print(first_val)

        # print(listOfCompanies[nameOfCompany])
        
        companySymbol = listOfCompanies[first_key]

        companyProfile = stockApiCall(companySymbol, 2)

        # companySymbol = listOfCompanies[nameOfCompany]

        # companyProfile = stockApiCall(companySymbol, 2)

        currentPrice = companyProfile[0]['price']

        # we got the price of the company using it's symbol
        print(f'the price of {companySymbol} is {currentPrice}')

        # retreiving quote from a certain time
        # quoteJson = stockApiCall(companySymbol, 3)
        returnVal = [companySymbol, first_key, currentPrice]

        return returnVal
    
        # print(quoteJson[0])
        # turn into a dataframe
        # df = pd.json_normalize(quoteJson)

        # print dataframe
        # print(df.loc['date'])

        # graph the data
        # graphData(df['date'].head(5),df['open'].head(5), nameOfCompany)

# a function to return a graph to a page based on the word that was searched
def name_to_graph(companySymbol):
    quoteJson = stockApiCall(companySymbol, 3)

    changeInPrice = getPriceChange(companySymbol)

    profileJson = stockApiCall(companySymbol, 1)
    
    profileDF = pd.json_normalize(profileJson)

    # print(profileDF)

    prices = pd.json_normalize(changeInPrice)

    df = pd.json_normalize(quoteJson)

    # Convert the date column to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Extract only the date part
    df['date'] = df['date'].dt.date

    # print(df['date'].tail(5))

    companyName = profileDF['name'][0]
    print(f"NAME IS {companyName}")
    res = graphData(df['date'],df['open'], companySymbol, prices, companyName)
    return res

#Setting up the database
conn=sqlite3.connect('personal-portfolio.db',check_same_thread=False)
c=conn.cursor()
c.execute('''
     CREATE TABLE IF NOT EXISTS portfolio (
     id Integer PRIMARY KEY,
     ticket TEXT NOT NULL,
     name TEXT NOT NULL,
     price Integer
     )
 ''')

#adds the stock onto the personal portfolio db
def addDatabase(ticket, name, price):
     # Check if the ticket already exists
     c.execute('SELECT * FROM portfolio WHERE ticket = ?', (ticket,))
     result = c.fetchone()

     if result:
         # Ticket already exists
         # do nothing
         return
     else:
         # New data and new ticket
         print("addDatabase")
         print(len(ticket))
         if len(ticket) > 0:

             c.execute('''
             INSERT INTO portfolio(ticket, name, price)
             VALUES (?, ?, ?)
             ''', (ticket, name, price))

#Setting up chatBot
'''
def chatBot():
    my_api_key=('OPENAI_KEY')

    client=OpenAI(
        api_key=my_api_key
        )
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a financial advisor or stock expert that gives advice to college students who are curious about anything related to stocks and financing for stocks"},
        {"role": "user", "content": "What are the advantages of pair programming?"}
    ]
 )
    print(completion.choices[0].message.content) 
'''

# my_api_key=('OPENAI_KEY')
# client=OpenAI(
#     api_key=my_api_key
# )

#Creating a chatbot response function. Gets the message from a request form from the frontend and 
#responds to the user input 
# @app.route('/chatbot',methods=['POST'])
# def chatBot():
#     #when the frontend is implemented, it would get the user message(or question) from a POST response
#     message=request.form.get("message")

#     completion = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": "You are a financial advisor or stock expert that gives advice to college students who are curious about anything related to stocks or finance"},
#         #Gets the user message
#         {"role": "user", "content":message}
#     ]
#     )

#     return completion.choices[0].message.content
  

#Creating a chatbot response function. Gets the message from a request form from the frontend and 
#responds to the user input 

# @app.route('/chatbot', methods=['POST'])
# def chatBot():
#     my_api_key = os.getenv('OPENAI_KEY')

#     client = OpenAI(
#         api_key=my_api_key,
#     )

#     try:
#             message = request.form.get("message")
#             print(f"Received message: {message}")  # Debugging statement

#             # Create a chat completion
#             completion = client.chat.completions.create(model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "user", "content": message}
#             ])

#             ai_response = completion.choices[0].message.content.strip()
#             print(f"AI Response: {ai_response}")  # Debugging statement

#             return jsonify({"response": ai_response})
#     except Exception as e:
#             print(f"Error: {e}")
#             return jsonify({"response": "Sorry, I didn't understand that."})

# getCompanyInfo()

# res = newsSearch("Microsoft")

# url = res[0]['url']
# print(url)

# sentimentAnalysis("https://www.cnbc.com/2024/07/16/self-proclaimed-bitcoin-inventor-craig-wright-referred-to-prosecutors.html")


app = Flask(__name__)

key = secrets.token_hex(16)

proxied = FlaskBehindProxy(app)

app.config['SECRET_KEY'] = key

# @app.route('/plot.png')
# def plot_png(symbol):
#     fig = name_to_graph(symbol)
#     output = io.BytesIO()
#     FigureCanvas(fig).print_png(output)
#     return Response(output.getvalue(), mimetype='image/png')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/portfolio', methods=['GET', 'POST'])
def portfolio():

    global stock_data
    form = addPortfolio()
    if form.validate_on_submit():
        if form.submit.data:
            # Handle the button click
            addDatabase(stock_data[0], stock_data[1], stock_data[2])
            tmp = c.execute('SELECT * FROM portfolio')
            html = tmp
            stock_data = ["", "", ""]
            print("in loop")
        return render_template('portfolio.html', table=html)
            

    print("outside area")
    tmp = c.execute('SELECT * FROM portfolio')
    html = tmp
    return render_template('portfolio.html', table=html)

"""
    form = userPrompt()

    if request.method == 'POST' and form.validate_on_submit():
        user_definition = form.getDefintion()
        word = form.word.data

        output = useChatGPT(str(user_definition),
                            word, getDefintion(word, uid, tokenid))

        form.word.data = getNewWord(word_list)
        addDatabase(output, word)
        return render_template('home.html', form=form,
                               message=output[0], grade=output[1],
                               word=form.word.data)

    form.word.data = getNewWord(word_list)
    return render_template('home.html', form=form, word=form.word.data)

"""
@app.route('/market', methods=['GET', 'POST'])

def market():
    """
    From the bottom of my heart, I hate this code with every fiber of my being
    I hope they use this in 300 years as a basis for a horror story
    Caught in a loop with no ***** ending, I love CS
    - Matthew
    """

    global stock_data
    form = userPrompt()
    add = addPortfolio()

    search = ""  

   
    if request.method == 'POST': 
        user_requested_company = form.getName()
        
        # print(user_requested_company)
        if len(user_requested_company) > 0 :


            stock_data = getCompanyInfo(user_requested_company)
            plot_html = name_to_graph(user_requested_company)
            return render_template('market.html', form=form, word=user_requested_company, add = add, ticket=stock_data[0], name=stock_data[1], value=stock_data[2])
            
            # url = "../static/images/new_plot.png"
            
    
    return render_template('market.html', form=form, add = add)
    
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

# Setting up chatBot
my_api_key = os.getenv('OPENAI_KEY')
openai.api_key = my_api_key

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
    app.run(debug=True,host="0.0.0.0", port=7000)
