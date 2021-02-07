import streamlit as st
import pickle
import pandas as pd
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
import zipfile

@st.cache(persist = True, suppress_st_warning = True)
def load_data():
	with open('./Data/articles.pkl', 'rb') as f:
		main_data = pickle.load(f)
	with zipfile.ZipFile('./Data/knn_model.zip', 'r') as zip_ref:
		zip_ref.extractall('./Data/knn/model')
	with zipfile.ZipFile('./Data/knn_vectorizer.zip', 'r') as zip_ref:
		zip_ref.extractall('./Data/knn/vectorizer')
	loaded_model = pickle.load(open('./Data/knn/model/knn_model.p','rb'))
	loaded_vectorizer = pickle.load(open('./Data/knn/vectorizer/knn_vectorizer.p','rb'))
	return main_data, loaded_model, loaded_vectorizer

st.title("Journalist Matching.")
st.markdown("Enter a news article to get matched with journos who covered similar stories in the past.")
st.text_area("News article goes here", "abcd")

data, model, vec = load_data()
st.write(data.head())