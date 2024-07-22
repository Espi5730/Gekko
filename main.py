from flask import Flask, render_template, request, jsonify
import requests
import re
from flask import Flask, render_template,jsonify,request, Response
from flask_behind_proxy import FlaskBehindProxy
from openai import OpenAI
from urllib.request import urlopen
from forms import userPrompt
from textblob import TextBlob
import secrets
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import certifi
import json
import requests
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
import matplotlib
import PyQt5
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.dates as mdates
from plotly.subplots import make_subplots
import plotly.graph_objs as go

app = Flask(__name__)

apiKey = "D5TvzaGcYfx4GOxOn834UD9QxCAyhEAH"
subscriptionKey = "f3f0023662b94a9cbfefa2b60472122e"
# apiKey="D5TvzaGcYfx4GOxOn834UD9QxCAyhEAH"
# subcriptionKey="f3f0023662b94a9cbfefa2b60472122e"

# functions
# function to get news
def newsSearch(searchTerm):
    search_url = "https://api.bing.microsoft.com/v7.0/news/search"
    headers = {"Ocp-Apim-Subscription-Key": subscriptionKey}
    params = {
        "q": f"{searchTerm} stock market finance",
        "textDecorations": True,
        "textFormat": "HTML",
        "mkt": "en-US",
        "count": 20  # to get more results and filter later
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
        # if a larger image URL is available in the API response so we can get a better quality photo
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
    fig.update_layout(title=f"{companyName}'s Prices", margin=dict(l=0, r=0, t=30, b=0))

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

    jsonOfCompanies = stockApiCall(nameOfCompany, 1)

    # print(jsonOfCompanies[0]['name'])

    listOfCompanies = {}

    for companyDict in jsonOfCompanies:
       
        resultName = companyDict['name']
        if resultName not in listOfCompanies:
            listOfCompanies[resultName] = companyDict['symbol']

    # list of companies that match what the user wanted 
    print(listOfCompanies)

    # now we are going to have the user choose what they wanted out of the list
    
    nameOfCompany = input("Pick one from the names provided \n")

    if nameOfCompany not in listOfCompanies:
        # edge case if the user picks something random

        print("bruh lock in")

        return
    else:

        # print(listOfCompanies[nameOfCompany])

        companySymbol = listOfCompanies[nameOfCompany]

        companyProfile = stockApiCall(companySymbol, 2)

        currentPrice = companyProfile[0]['price']

        # we got the price of the company using it's symbol
        print(f'the price of {companySymbol} is {currentPrice}')

        # retreiving quote from a certain time
        quoteJson = stockApiCall(companySymbol, 3)

        # print(quoteJson[0])
        # turn into a dataframe
        df = pd.json_normalize(quoteJson)

        # print dataframe
        # print(df.loc['date'])

        # graph the data
        graphData(df['date'].head(5),df['open'].head(5), nameOfCompany)

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
    name TEXT NOT NULL,
    price Integer,
    changeInPrice Integer
    )
''')

#adds the stock onto the personal portfolio db
def addDatabase(name,price,changeInPrice):
    c.execute('''
    INSERT INTO portfolio(name,price,changeInPrice),
    VALUES (?,?,?)
    ''',(name,price,changeInPrice)
    )

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

my_api_key=('OPENAI_KEY')
client=OpenAI(
    api_key=my_api_key
)

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

#random comment



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

@app.route('/portfolio')
def portfolio():
    return render_template('portfolio.html')
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
    form = userPrompt()

    search = ""  

    # autocomplete?

    # jsonOfCompanies = getAllCompanies()

    # print(jsonOfCompanies[0]['name'])

    # listOfCompanies = {}

    # for companyDict in jsonOfCompanies:
       
    #     resultName = companyDict['name']
    #     if resultName not in listOfCompanies:
    #         listOfCompanies[resultName] = companyDict['symbol']

    # # print(listOfCompanies["Perth Mint Gold"])

    # data = listOfCompanies.items()

    if request.method == 'POST': 
        user_requested_company = form.getName()
        
        # print(user_requested_company)
        if len(user_requested_company) > 0 :  
            plot_html = name_to_graph(user_requested_company)
            # url = "../static/images/new_plot.png"
            

            

            return render_template('market.html', form=form, word=user_requested_company, plot_html = plot_html)
    
    return render_template('market.html', form=form,)
    

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
    app.run(debug=True,host="0.0.0.0", port=5000)
