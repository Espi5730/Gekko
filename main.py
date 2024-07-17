from flask import Flask, render_template
from urllib.request import urlopen
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import certifi
import json



apiKey="D5TvzaGcYfx4GOxOn834UD9QxCAyhEAH"
# functions

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

        
        graphData(df['date'].head(5),df['open'].head(5), nameOfCompany)
















getCompanyInfo()

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')



# if __name__ == '__main__':
#     app.run(debug=True)
