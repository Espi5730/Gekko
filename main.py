from flask import Flask, render_template
import sqlite3
from urllib.request import urlopen
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import certifi
import json
import requests
from textblob import TextBlob
# from newspaper import Article
# from newspaper import Config

import newspaper



apiKey="D5TvzaGcYfx4GOxOn834UD9QxCAyhEAH"
subcriptionKey="f3f0023662b94a9cbfefa2b60472122e"
# functions

# function to search for news on a company 
def newsSearch(searchTerm):
    search_url = "https://api.bing.microsoft.com/v7.0/news/search"

    headers = {"Ocp-Apim-Subscription-Key" : subcriptionKey}
    params  = {"q": searchTerm, "textDecorations": True, "textFormat": "HTML"}

    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    results = response.json()

    # this will get you the first url
    # print(results['value'][0]['url'])

    # a list of the news articles from the searched word
    stories = results['value']

    resultList = []

    for story in stories:

        resultList.append( { 'name' : story['name'], 'url' : story['url'], 'image' : story['image']['thumbnail']['contentUrl'], 'description' : story['description'], 'provider' : story['provider'][0]['name'], 'data' : story['datePublished']} )
    
    # print(resultList[0])

    return resultList


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
#     article = newspaper.article('https://www.msn.com/en-us/news/technology/microsofts-ai-powered-designer-app-is-out-of-preview-mode/ar-BB1qdWBb')
#     article.download()
#     article.parse()
#     article.nlp()

#     text = article.summary
#     print(article.title)
#     blob = TextBlob(text)

#     sentiment = blob.sentiment.polarity

#     print(sentiment)

# function specifically made to make an api call
def stockApiCall(nameOfCompany, option):
    if option == 1:
        # use api to return a list of company names that match the name 

        generalSearchUrl = (f'https://financialmodelingprep.com/api/v3/search?query={nameOfCompany}&limit=10&&apikey={apiKey}')
        
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
        companyHistoryPriceUrl = (f'https://financialmodelingprep.com/api/v3/historical-chart/30min/{nameOfCompany}?from=2023-09-10&to=2023-09-11&apikey={apiKey}')

        # turn the request into json format

        response = urlopen(companyHistoryPriceUrl, cafile=certifi.where())
        
        data = response.read().decode("utf-8")
        
        jsonOfCompanies = json.loads(data)
        
        return jsonOfCompanies
    
# function to make line graph from the time and quotes of a company
def graphData(independant, dependant, companyName):

    

    plt.rc('font', size=4)    # font size

    x = np.array(independant)

    y = np.array(dependant)

    plt.plot(x,y)

    plt.xlabel("Dates")  # add X-axis label
    plt.ylabel("Price")  # add Y-axis label
    plt.title(f"{companyName}'s prices")  # add title
   

    plt.show()

    
# function to find stock information on comapny
def getCompanyInfo():

    # ask user what company they want to see
    nameOfCompany = str(input("What is the name of the company you want to look up?\n"))

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



# getCompanyInfo()

# res = newsSearch("Microsoft")

# url = res[1]['url']
# print(url)



app = Flask(__name__)

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

#adds the stock onto a personal portfolio db
def addDatabase(name,price,changeInPrice):
    c.execute('''
    INSERT INTO portfolio(name,price,changeInPrice),
    VALUES (?,?,?)
    ''',(name,price,changeInPrice)
    )

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0")
