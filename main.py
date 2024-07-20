from flask import Flask, render_template,jsonify,request
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

    if request.method == 'POST': 
        user_definition = form.getName()
        
        print(user_definition)
        if len(user_definition) > 0 :  
           





            return render_template('market.html', form=form, word=user_definition)
    
    return render_template('market.html', form=form,)
    

@app.route('/resources')
def learn():
    return render_template("resources.html")











if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0", port=5001)
