import pickle
import pandas as pd
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

authors = pd.read_csv('../Data/journalists_v2.csv')
articles = pd.read_csv('../Data/articles_v2.csv')

vectorizer = TfidfVectorizer(stop_words = 'english')
X = vectorizer.fit_transform(list(articles.content))

nn = NearestNeighbors(n_neighbors=5, radius=1.0, algorithm='brute', leaf_size=30, metric='cosine', p=2, metric_params=None, n_jobs=None)
nn.fit(X)

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}

# Cleaning the text sentences so that punctuation marks, stop words &amp; digits are removed
# Words are lemmatized according to their POS tags
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
    input_text = [clean(text)]
    new = vectorizer.transform(input_text)
    results = nn.kneighbors(new.todense())
    for i in range(5):
        index = results[1][0][i]
        print(articles.iloc[index].author_name_clean.title())
#         print(data.iloc[index].full_text)
        print(articles.iloc[index].topic)
        print(articles.iloc[index].site_name)
        print(articles.iloc[index].full_text)
        print(humanize.naturaltime(datetime.datetime.now() - datetime.datetime(*map(int, articles.date[index].split('-')))))
        print()
    return results

print(similar_journalists("When everyone is fit in both squads? I think it’s very difficult to say, they’re both outstanding at creating and scoring goals, you probably couldn’t split them. But at the moment Liverpool certainly have the upper hand."))

print(similar_journalists("There are as many as 55 long-term effects of Covid-19 that affect the body, with fatigue being the most common symptom among recovered patients, according to researchers who reviewed existing studies on the virus. For a study, published in the journal Scientific Reports, researchers screened as many as 18,251 publications, out of which 15 studies were selected for final analysis. An international team of researchers included eight studies from Europe and the UK, three from the US and one each from Australia, China, Egypt and Mexico for the study, published on 9 August. The number of patients that were followed up in the studies ranged from 102 to 44,799. Adults ranging from 17 to 87 years of age were included in them and the patient follow-up time ranged from 14 to 110 days. Ten studies collected information from the patients using self-reported surveys. Two studies collected data from medical records and three by clinical evaluation. Six out of the 15 studies included only patients hospitalised for Covid-19 while the rest of the studies used mixed data from mild, moderate and severe Covid-19 patients. In total, the study identified as many as 55 long-term effects associated with Covid-19 and most of them corresponded to clinical symptoms such as fatigue, headache, complete or partial loss of smell (anosmia) and joint pain. However, along with these, diseases such as stroke and diabetes mellitus — metabolic disorders that cause high blood sugar levels — were also discovered."))

pickle.dump(nn, open('../Models/knn_model.p','wb'))
pickle.dump(vectorizer, open("../Models/knn_vectorizer.p", "wb"))

# get tfidf scores at ith index and for top n words

def get_tfidf_weights(i, n):

    vector = X[i] 
    df = pd.DataFrame(vector.T.todense(), index = vectorizer.get_feature_names(), columns = ["tfidf"]) 
    df = df.sort_values(by = ["tfidf"], ascending=False)
    df = df[df.tfidf > 0].reset_index()
    df.columns = ['Word', 'TF - IDF']
    
    print("Title:", articles.iloc[i].title)
    print("Description:", articles.iloc[i].description)
    print("Full text", articles.iloc[i].full_text)
    
    return df.head(n)

fig = px.bar(get_tfidf_weights(0, 15), x = "Word", y = "TF - IDF")
fig.update_xaxes(tickangle = 90)
fig.show()