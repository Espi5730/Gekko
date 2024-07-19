from flask import Flask, render_template

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

@app.route('/news')
def learn():
    return render_template("news.html")
    
if __name__ == '__main__':
    app.run(debug=True)
