import pickle
import re
import pandas as pd
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import wordnet
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

iab_text = pd.read_csv('../Data/IAB_text_v3.csv')
iab_text.head()

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

df = iab_text.dropna(subset = ['with_justext_clean'])
df = df[df.with_justext_clean != 'exceed'].reset_index(drop = True)

tfidf = TfidfVectorizer()
text_count = tfidf.fit_transform(df['with_justext_clean'])
y = np.asarray(df[df.columns[7:11]])

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(text_count, y, test_size = 0.25, random_state = 5)

print(len(mlb.classes_))

knnClf = KNeighborsClassifier(n_neighbors = 3)
knnClf.fit(x_train, y_train)
knnpred = knnClf.predict(x_test)

print('accuracy_score_KNN', metrics.accuracy_score(knnpred, y_test))
print('recall_macro_score', metrics.recall_score(knnpred, y_test,average = 'macro', zero_division = 1))
print('recall_micro_score', metrics.recall_score(knnpred, y_test,average = 'micro', zero_division = 1))
print('recall_weighted_score', metrics.recall_score(knnpred, y_test,average = 'weighted', zero_division = 1))
print('f1_macro_score', metrics.f1_score(knnpred, y_test,average = 'macro', zero_division = 1))
print('f1_micro_score', metrics.f1_score(knnpred, y_test,average = 'micro', zero_division = 1))
print('f1_weighted_score', metrics.f1_score(knnpred, y_test,average = 'weighted', zero_division = 1))
print('precision_macro_score', metrics.precision_score(knnpred, y_test,average = 'macro', zero_division = 1))
print('precision_micro_score', metrics.precision_score(knnpred, y_test,average = 'micro', zero_division = 1))
print('precision_weighted_score', metrics.precision_score(knnpred, y_test,average = 'weighted', zero_division = 1))

s = "Jump to navigation NEWSLIVE TV Home APPMAGAZINE HOMEMY FEEDVIDEOSMALAYALAMINDIAGAMINGFACT CHECKQUIZMOVIESHEALTHTECHSPORTSDIU NewsLifestyleYoga can help you keep heart diseases at bay Yoga can help you keep heart diseases at bay Yoga can become a way of life to help you deal with heart diseases. Priyanka Sharma New Delhi June 21, 2018UPDATED: June 21, 2018 18:59 IST Chair posture is proving beneficial for patients who can’t sit on the floor or have disabilities Chair posture is proving beneficial for patients who can’t sit on the floor or have disabilities Even in matters of the heart, literally speaking, yoga has an impact. A study by the All India Institute of Medical Sciences (AIIMS) has established that a person can regulate her/his heartbeats through yoga. AIIMS, in what it claimed to be the first study of its kind to have scientifically established the benefits of yoga, revealed that breathing in a particular rhythm has many positive effects on cardiovascular health. It also helps in overcoming anger, fear, stress and hypertension. Dr KK Deepak, author of the study and head of the Department of Physiology at AIIMS, told Mail Today: A slow yogic breathing can curtail the feeling of stress, fear, anger, tension and also regulate diabetes. The study has found that there is a correlation between heartbeat and blood pressure. Yogic breathing can synchronise the two which is a big deal.  The results of the study, funded by Indian Council for Medical Research (ICMR) and the Ministry of AYUSH, have been published in the latest issue of Indian Journal of Medical Research. When we do deep-breathing, blood pressure comes into the normal range. This indicates that there is a strong relationship between heart rate and blood pressure. If a persons heart rate falls, suddenly the BP will go up and if the BP comes down, the heart rate shoots up. Thus, creating a balance between the two is very important - which can be done by yogic breathing,  Dr Deepak said. In another study conducted by RML Hospital, doctors have found that practising asanas led to significant improvement in patients suffering from mental health illnesses, as compared with those who did other forms of exercise. We made two groups of patients yoga and non-yoga. The yoga group patients cognitive behaviour improved tremendously as compared to the patients doing other exercises, said (Prof) Smita N. Deshpande, Head the Department of Psychiatry & De-Addiction at RML Hospital. Chair Yoga for health Offering relief to patients who cannot sit on floor, Delhi's Sir Ganga Ram Hospital has introduced a new yoga initiative called 'chair yoga' in which the patients can remain seated on a chair and perform health postures. Dr Soina Rawat, director at the department of executive health check up at the hospital, said,A large number of patients were facing problems in doing yoga because of various disabilities. They wanted to practice yoga but couldnt sit on the floor. So we started the concept of chair yoga at our hospital. It is really helpful and their health status has improved. Binda Shukla, 50, is one patient who has been practicing chair yoga for several months. She suffers from upper body stiffness, and has cervical and cholesterol problems. There is definitely relief ever since I started doing meditation and yoga. I feel fresh when I perform deep breathing asanas and am taking less medication now. Even those who have knee joint related complications are comfortable with chair yoga, says Dr Shukla To see whether yoga can improve cognitive functions of patients with severe mental disorder as compared to cardiac controls and to compare improvements in patients with schizophrenia. Time period: Samples of last two years in the study included patients with depression, bipolar disorder, schizophrenia and cardiac controls. Patients were aged between 18-60. Another study was conducted to ascertain randomised control trail on schizophrenia on yoga and non-yoga groups. Exercises: Yoga training included chanting of Om, warm up exercises, breathing (pranayama), various yogic asanas Results: There was an improvement in the speed parameters of most of the cognitive domains among patients with schizophrenia after yoga, compared with the schizophrenia non-yoga group patients. The healthy parameters sustained for six months at least. Tags :Follow YogaFollow International Yoga Day POST A COMMENT READ THIS BJP chief JP Nadda Top BJP leaders take to social media to condemn tweets on farmers' stir by Rihanna, Greta The efficacy of double masking: What health experts have to say Silenced Minority? Airtel, Jio, Vi best prepaid plans with streaming and data benefits under Rs 500 RECOMMENDED WATCH RIGHT NOW 01:00 Watch: Arvind Kejriwal takes first dose of Covid vaccine 02:27 Bengal assembly polls: Mamata Banerjee to file nomination from Nandigram on March 11 00:49 Bengali singer Aditi Munshi joins Trinamool Congress 02:57 Karnataka minister Ramesh Jarkiholi resigns after sex CD scandal 02:01 Good news: Prisoners beautify jail compound in UP's Maharajganj TOP TAKES EC directs petrol pumps to remove hoardings showing PM Modi's photos01:10 EC directs petrol pumps to remove hoardings showing PM Modi's photos BJP calls Rahul Gandhi's push-up challenge violation of code of conduct, writes to EC00:46 BJP calls Rahul Gandhi's push-up challenge violation of code of conduct, writes to EC Delhi Police foils plan to kill 2 Delhi riots accused in Tihar Jail02:30 Delhi Police foils plan to kill 2 Delhi riots accused in Tihar Jail INDIATODAY.IN Download App Andriod AppIOS AppSmartTv App Copyright © 2021 Living Media India Limited. For reprint rights: Syndications Today Covid-19 pandemic: 5 things one should know about heart failure Covid-19 pandemic: 5 things one should know about heart failure KCR's Backward Classes Challenge KCR's Backward Classes Challenge What genes tell us about the risk of developing cancer What genes tell us about the risk of developing cancer Redefining heart health with cutting edge technologies Redefining heart health with cutting edge technologies How yoga can help you to get rid of stress headache How yoga can help you to get rid of stress headache Designer sarees to mark your distinctive impression Designer sarees to mark your distinctive impression Designer sarees to mark your distinctive impression Designer sarees to mark your distinctive impression Do's and dont's to keep in mind while opting for IVF treatment: Here's all you need to know Do's and dont's to keep in mind while opting for IVF treatment: Here's all you need to know Be a style icon with these groovy women handbags Be a style icon with these groovy women handbags India's Avant-Garde India's Avant-Garde"
s = clean(s)
s = tfidf.transform([s])
print(mlb.inverse_transform(knnClf.predict(s)))

f = open('../Models/IAB_binarizer.p', 'wb')
pickle.dump(mlb, f)
f.close()

f = open('../Models/IAB_vectorizer.p', 'wb')
pickle.dump(tfidf, f)
f.close()

f = open('../Models/IAB_classifier.p', 'wb')
pickle.dump(knnClf, f)
f.close()

import sklearn
sklearn.__version__