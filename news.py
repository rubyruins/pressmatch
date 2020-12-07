import pandas as pd
import numpy as np
import json
import ftfy
import re
from afinn import Afinn
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

sources = ['The Indian Express', 'News | DW', 'Express.co.uk', 'Washington Times', 'Mirror Online', 'postandcourier.com', 'richmond.com', 'Baltimore Sun', 'Wales Online', 'Daily Star', 'The Financial Express', 'Daily Mail Online', 'Business Insider', 'CNN Underscored', 'CNN', 'unionleader.com', 'Digg', 'journalstar.com', 'Seeking Alpha', 'madison.com', 'Chicago Tribune', 'The Morning Call', 'Heavy.com', 'Daily Press', 'Vanity Fair', 'Variety', 'Japan Today', 'omaha.com']
lemmatizer = WordNetLemmatizer()
afinn = Afinn(language='en')

def clean_title(text):
	for source in sources:
		if source in text:
			text = text.replace(source, '')
	return text

def clean(text):
	text = ftfy.fix_text(text)
	text = re.sub(r'\n',r'',text)
	text = re.sub(r'[^A-Za-z\s]',r'',text)
	tokens = nltk.word_tokenize(text)
	tokens = [word.lower() for word in tokens]
	stop_words = stopwords.words('english')
	tokens = [token for token in tokens if token not in stop_words]
	tokens = [lemmatizer.lemmatize(token, pos="v") for token in tokens]
	text = ' '.join([str(token) for token in tokens]) 
	return text

def sentiment_score(text):
	return round(afinn.score(text) / len(text.split()) * 100, 2)

@st.cache(persist = True)
def load_data():
	data = pd.DataFrame(columns = ['author', 'content', 'description', 'publishedAt', 'source_name', 'source_url', 'title', 'url', 'urlToImage']) 
	for i in range(1, 6):
		with open(f'data/{i}.json', 'r', encoding = "utf8") as j:
			contents = json.loads(j.read())
			contents = pd.json_normalize(contents['Article'])
			data = pd.concat([data, contents], ignore_index=True).drop_duplicates()
	data.pop('source_url')
	data.pop('urlToImage')
	data.pop('publishedAt')
	data.pop('author')
	data.pop('description')
	data.pop('url')
	data['clean_title'] = data['title'].apply(lambda x: clean_title(x))
	data['clean_title'] = data['clean_title'].apply(lambda x: clean(x))
	data['clean_content'] = data['content'].apply(lambda x: clean(x))
	data['content_score'] = data['content'].apply(lambda x: sentiment_score(x))
	return data

@st.cache()
def NER():
	all_entities = pd.read_csv("data/Entities.csv")
	all_entities.pop('Unnamed: 0')
	return all_entities

@st.cache()
def coverage(term):
	temp = df[df['content'].str.contains(f'(?i){term}')].groupby(by = 'source_name')['content_score'].agg('mean').reset_index().sort_values(by = 'content_score') 
	temp.replace('', np.nan, inplace=True)
	temp.dropna(inplace=True)
	temp = temp.rename(columns = {'content_score': 'Sentiment', 'source_name': 'Source'})
	return temp

@st.cache()
def trending(term):
	entities = dict()
	for sent in nltk.sent_tokenize(' '.join(df[df['content'].str.contains(f'(?i){term}')].content.values)):
		for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
			if hasattr(chunk, 'label'):
				tag = chunk.label()
				val = ' '.join(c[0] for c in chunk)
				if tag in entities:
					if val in entities[tag]:
						entities[tag][val] += 1
					else:
						entities[tag][val] = 1
				else:
					entities[tag] = dict()
					entities[tag][val] = 1
	all_entities = pd.DataFrame(columns = ['Tag', 'Entity', 'Count'])
	for tag in entities:
		for item in entities[tag]:
			row = {'Tag': tag, 'Entity': item, 'Count': entities[tag][item]}
			all_entities = all_entities.append(row, ignore_index = True) 
	all_entities = all_entities.sort_values('Count', ascending = False)
	temp = all_entities[((all_entities.Tag == 'ORGANIZATION') | (all_entities.Tag == 'PERSON'))]
	temp = temp[temp.Entity != term]
	return temp

st.title("News analysis.")
st.markdown("Identifying entities in news articles with a sentiment analysis of their coverage by various outlets. Get started by selecting entity options from the sidebar.")

df = load_data()
ner = NER()

st.subheader("Topics trending today.")
st.sidebar.subheader("Options.")
entries = st.sidebar.slider('Number of topics', min_value = 10, max_value = 100)
tags = st.sidebar.multiselect('Entity types', ['ORGANIZATION', 'PERSON'], default = ['PERSON'])

temp = ner[((ner.Tag.isin(tags)))].head(entries).reset_index()
st.plotly_chart(px.line(temp, x = 'Entity', y = 'Count', facet_col = 'Tag').update_xaxes(matches=None))
st.write(f"{', '.join(temp.iloc[i]['Entity'] for i in range(5))} seem to be trending today. Here's a wordcloud to put things into perspective.")
words = dict()
for i in range(len(temp)):
	words[temp.iloc[i]['Entity']] = temp.iloc[i]['Count']
wc = WordCloud(stopwords = stopwords.words('english'), background_color='white')
ner_wordcloud = wc.generate_from_frequencies(words)
st.image(ner_wordcloud.to_array(), width=600)

st.subheader("Select an entity to analyse their recent mentions in the news.")
entity = st.sidebar.selectbox('Select an entity', temp.Entity.unique())
st.write(f"How did news outlets cover {entity}?")
sources = coverage(entity)
sources = sources[sources.notnull().all(axis = 1)]
st.plotly_chart(px.bar(sources, x = 'Source', y = 'Sentiment', color = 'Sentiment').update_layout(height = 400))
st.write(f"Trending along with {entity}...")
temp = trending(entity).head(10)
temp.Count = pd.to_numeric(temp['Count'])
st.plotly_chart(px.scatter(temp, x = 'Entity', y = 'Count', size = 'Count'))