from flask import Flask, render_template,jsonify
import os
from openai import OpenAI
from flask import request



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

@app.route('/resources')
def learn():
    return render_template("resources.html")






#Creating a chatbot response function. Gets the message from a request form from the frontend and 
#responds to the user input 

@app.route('/chatbot', methods=['POST'])
def chatBot():
    my_api_key = os.getenv('OPENAI_KEY')

    client = OpenAI(
        api_key=my_api_key,
    )

    try:
            message = request.form.get("message")
            print(f"Received message: {message}")  # Debugging statement

            # Create a chat completion
            completion = client.chat.completions.create(model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": message}
            ])

            ai_response = completion.choices[0].message.content.strip()
            print(f"AI Response: {ai_response}")  # Debugging statement

            return jsonify({"response": ai_response})
    except Exception as e:
            print(f"Error: {e}")
            return jsonify({"response": "Sorry, I didn't understand that."})




if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0", port=5001)
