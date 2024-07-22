from flask import Flask, render_template, request, jsonify
from flask import Flask, render_template,jsonify,request, Response
from flask_behind_proxy import FlaskBehindProxy
from openai import OpenAI
from urllib.request import urlopen
from forms import userPrompt
from textblob import TextBlob
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from plotly.subplots import make_subplots
from add import addPortfolio
import requests
import re
import secrets
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import certifi
import json
import requests
import os
import io
import matplotlib
import PyQt5
import matplotlib.dates as mdates
import plotly.graph_objs as go


apiKey = "D5TvzaGcYfx4GOxOn834UD9QxCAyhEAH"
subscriptionKey = "f3f0023662b94a9cbfefa2b60472122e"
# apiKey="D5TvzaGcYfx4GOxOn834UD9QxCAyhEAH"
# subcriptionKey="f3f0023662b94a9cbfefa2b60472122e"

#validators
stock_data = ["", "", ""]

# functions
# function to get news

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
    returnVal = ["", "", ""]
    try:  
        jsonOfCompanies = stockApiCall(nameOfCompany, 1)
    except:
        return [" "," "," "]


    # print(jsonOfCompanies[0]['name'])

    listOfCompanies = {}

    for companyDict in jsonOfCompanies:
       
        resultName = companyDict['name']
        if resultName not in listOfCompanies:
            listOfCompanies[resultName] = companyDict['symbol']

    # list of companies that match what the user wanted 

    if listOfCompanies is None:
        return [" "," "," "]

    # now we are going to have the user choose what they wanted out of the list
    
    #nameOfCompany = input("Pick one from the names provided \n")

    if listOfCompanies is None:
        # edge case if dict is empty
        print("bruh lock in")
        returnVal = ["", "", ""]
        return returnVal
    else:
        
        print(listOfCompanies)
        first_key = list(listOfCompanies)[0]
        first_val = list(listOfCompanies.values())[0]
        print(first_key)
        print(first_val)

        # print(listOfCompanies[nameOfCompany])

        companySymbol = listOfCompanies[first_key]

        companyProfile = stockApiCall(companySymbol, 2)

        currentPrice = companyProfile[0]['price']

        # we got the price of the company using it's symbol
        print(f'the price of {companySymbol} is {currentPrice}')

        returnVal = [companySymbol, first_key, currentPrice]

        return returnVal


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
    c.execute('''
    INSERT INTO portfolio(ticket, name, price)
    VALUES (?,?,?)
    ''',(ticket, name, price)
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

  
# functions
# function to get news
def newsSearch(searchTerm):
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

    graph_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    return f'<div style="width:50%;">{graph_html}</div>'


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


#Set up flask

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


@app.route('/portfolio', methods=['GET', 'POST'])
def portfolio():
    global stock_data


    if stock_data[0] != "": 
        addDatabase(stock_data[0], stock_data[1], stock_data[2])
        tmp = c.execute('SELECT * FROM portfolio')
        html = tmp
        stock_data = ["", "", ""]
        print("in loop")

        return render_template('portfolio.html', table=html)

    #addDatabase(stock_data[0], stock_data[1], stock_data[2])
    tmp = c.execute('SELECT * FROM portfolio')
    html = tmp

    print("out of loop")
    #print(html)
    #print("after")

    return render_template('portfolio.html', table=html)

@app.route('/market', methods=['GET', 'POST'])
def market():
    global stock_data
    
    form = userPrompt()
    add = addPortfolio()
    search = ""  

    if request.method == 'POST': 
        user_definition = form.getName()
        user_requested_company = form.getName()

        #if add portfolio clicked 
        print(user_definition)
        if len(user_definition) > 0 : 
            #api requests

            stock_data = getCompanyInfo(user_definition)
            plot_html = name_to_graph(user_requested_company)


            print(stock_data)
            #show graph (brice)


            #display api results on page
            return render_template('market.html', form=form, add=add, ticket=stock_data[0], name=stock_data[1], value=stock_data[2],  plot_html = plot_html)

    return render_template('market.html', form=form, add=add)
    

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
    app.run(debug=True,host="0.0.0.0", port=5001)
