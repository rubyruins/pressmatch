from flask import *
import pandas as pd
from scipy import stats
import numpy as np
import pickle
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re
import ftfy
import humanize
import datetime
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import plotly.express as px

articles = pd.read_csv('./Data/articles_v2.csv')
knn_model = pickle.load(open('./Models/knn_model.p','rb'))
knn_vec = pickle.load(open('./Models/knn_vectorizer.p', 'rb'))
iab_classifier = pickle.load(open('./Models/IAB_classifier.p','rb'))
iab_binarizer = pickle.load(open('./Models/IAB_binarizer.p','rb'))
iab_vectorizer = pickle.load(open('./Models/IAB_vectorizer.p','rb'))

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}

def clean(doc):
	doc = ftfy.fix_text(doc)
	stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
	punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
	normalized = ""
	for word in punc_free.split():
		tag = nltk.pos_tag([word])[0][1][0].upper()
		normalized = normalized + " " + str(lemma.lemmatize(word, tag_dict.get(tag, wordnet.NOUN)))
	processed = re.sub(r"\d+","", normalized)
	y = processed.split()
	return ' '.join(y)

def similar_journalists(text):
	input_text = clean(text)
	new = knn_vec.transform([input_text])
	results = knn_model.kneighbors(new.todense())
	
	data = []
	for i in range(5):
		news_item = dict()
		index = results[1][0][i]
		tags = list(iab_binarizer.inverse_transform(iab_classifier.predict(iab_vectorizer.transform([clean(articles.iloc[index].full_text)])))[0])
		tags = ', '.join([i for i in tags if i != '-'])
		
		news_item['author_name'] = articles.iloc[index].author_name_clean.title()
		news_item['topic'] = articles.iloc[index].topic
		news_item['site_name'] = articles.iloc[index].site_name
		news_item['full_text'] = articles.iloc[index].full_text
		news_item['time'] = humanize.naturaltime(datetime.datetime.now() - datetime.datetime(*map(int, articles.date[index].split('-'))))
		news_item['tags'] = tags
		data.append(news_item)
		
		# print(data)
	return data

app = Flask(__name__)

@app.route('/')
def home():
	return render_template("home.html")

@app.route('/results', methods =['POST'])
def basic():
	input_article = request.form['input_article']
	data = similar_journalists(input_article)
	return render_template("display.html", results = data)

@app.errorhandler(404)
def error(e):
	return render_template("notfound.html")

if __name__ == "__main__":
	app.run(debug = True)