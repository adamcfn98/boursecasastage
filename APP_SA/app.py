from flask import Flask, render_template, request, jsonify
from scripts import analyser
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyse', methods=['POST'])
def analyse():
    if request.method == 'POST':
        stock_name = request.form['stock_name']
        sentiment_scores, titles = analyser.analyser(stock_name)
        print("Sentiment Scores:", sentiment_scores)  
        return render_template('results.html', sentiment_scores=sentiment_scores, stock_name=stock_name, titles=titles)

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    titles = request.json['titles']
    text = ' '.join(titles)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    img = BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode()
    return jsonify({'img_base64': img_base64})

if __name__ == '__main__':
    app.run(debug=True)

