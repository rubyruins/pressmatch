import streamlit as st
import pickle
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re
import ftfy
import humanize
import datetime
import zipfile
import shutil
import copy

@st.cache(persist = True, suppress_st_warning = True, max_entries = 3)
def load_data():
	with zipfile.ZipFile('./Data/articles.zip', 'r') as zip_ref:
		zip_ref.extractall('./Data/knn/articles')
	with zipfile.ZipFile('./Data/knn_model.zip', 'r') as zip_ref:
		zip_ref.extractall('./Data/knn/model')
	with zipfile.ZipFile('./Data/knn_vectorizer.zip', 'r') as zip_ref:
		zip_ref.extractall('./Data/knn/vectorizer')
	main_data = pickle.load(open('./Data/knn/articles/articles.pkl','rb'))
	model = pickle.load(open('./Data/knn/model/knn_model.p','rb'))
	vec = pickle.load(open('./Data/knn/vectorizer/knn_vectorizer.p','rb'))
	shutil.rmtree('./Data/knn')
	return main_data, model, vec

# Cleaning the text sentences so that punctuation marks, stop words &amp; digits are removed
def clean(doc):
	doc = ftfy.fix_text(doc)
	stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
	punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
	normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
	processed = re.sub(r"\d+","", normalized)
	y = processed.split()
	return ' '.join(y)

def similar_journalists(text):
	input_text = [clean(text)]
	new = vec.transform(input_text)[:]
	results = model.kneighbors(new.todense())[:]
	return results

nltk.download('stopwords')
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
data, model, vec = copy.deepcopy(load_data())

st.title("Journalist Matching.")
st.markdown("Enter a news article to get matched with journos who covered similar stories in the past.")
to_match = st.text_area("News article goes here", "Type something...")
show_full = st.checkbox("Show full article text?")
results = similar_journalists(clean(to_match))
for i in range(5):
	index = results[1][0][i]
	st.write(f"{humanize.naturaltime(datetime.datetime.now() - datetime.datetime(*map(int, data.date[index].split('-'))))}: {data.iloc[index].clean_author.title()} in {data.iloc[index].topic} for {data.iloc[index].site_name}")
	if show_full:
		st.write(data.iloc[index].full_text)
		st.write()

# When everyone is fit in both squads? I think it’s very difficult to say, they’re both outstanding at creating and scoring goals, you probably couldn’t split them. But at the moment Liverpool certainly have the upper hand.