from flask import Flask, render_template
import os 
import openai
from openai import OpenAI

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

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





if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0")
