from flask import Flask, render_template
import sqlite3

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

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0")
