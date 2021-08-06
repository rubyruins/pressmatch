import re
import nltk
import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess(text):
    if isinstance((text), (str)):
        text = re.sub('<[^>]*>', ' ', text)
        text = re.sub('[\W]+', ' ', text.lower())
        return text
    if isinstance((text), (list)):
        return_list = []
        for i in range(len(text)):
            temp_text = re.sub('<[^>]*>', '', text[i])
            temp_text = re.sub('[\W]+', '', temp_text.lower())
            return_list.append(temp_text)
        return(return_list)

data = pd.read_csv("./Data/IAB/iab_text_tiers.csv")
stop = stopwords.words('english')

df = data[data['with_bs4'].notna()]
df = df[df['with_bs4'] != "exceeded"]
df['with_bs4'] = df['with_bs4'].apply(preprocess)
df = df[df['with_justext'].notna()]
df = df[df['with_justext'] != "exceeded"]
df['with_bs4'] = df['with_justext'].apply(preprocess)
df = df.reset_index()
print(len(df))

tfidf = TfidfVectorizer(min_df = 5, max_df = 0.5)
text_count = tfidf.fit_transform(df['with_bs4'])
y = np.asarray(df[df.columns[9:13]])

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(text_count, y, test_size = 0.25, random_state = 5)

knnClf = KNeighborsClassifier(leaf_size = 1, n_neighbors = 3)
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

s = "A study by the All India Institute of Medical Sciences (AIIMS) has established that a person can regulate her/his heartbeats through yoga. AIIMS, in what it claimed to be the first study of its kind to have scientifically established the benefits of yoga, revealed that breathing in a particular rhythm has many positive effects on cardiovascular health. It also helps in overcoming anger, fear, stress and hypertension. Dr KK Deepak, author of the study and head of the Department of Physiology at AIIMS, told Mail Today: A slow yogic breathing can curtail the feeling of stress, fear, anger, tension and also regulate diabetes. The study has found that there is a correlation between heartbeat and blood pressure. Yogic breathing can synchronise the two which is a big deal.  The results of the study, funded by Indian Council for Medical Research (ICMR) and the Ministry of AYUSH, have been published in the latest issue of Indian Journal of Medical Research. When we do deep-breathing, blood pressure comes into the normal range. This indicates that there is a strong relationship between heart rate and blood pressure. If a persons heart rate falls, suddenly the BP will go up and if the BP comes down, the heart rate shoots up. Thus, creating a balance between the two is very important - which can be done by yogic breathing,  Dr Deepak said. In another study conducted by RML Hospital, doctors have found that practising asanas led to significant improvement in patients suffering from mental health illnesses, as compared with those who did other forms of exercise. We made two groups of patients yoga and non-yoga. The yoga group patients cognitive behaviour improved tremendously as compared to the patients doing other exercises, said (Prof) Smita N. Deshpande, Head the Department of Psychiatry & De-Addiction at RML Hospital. Chair Yoga for health Offering relief to patients who cannot sit on floor, Delhi's Sir Ganga Ram Hospital has introduced a new yoga initiative called 'chair yoga' in which the patients can remain seated on a chair and perform health postures. Dr Soina Rawat, director at the department of executive health check up at the hospital, said,A large number of patients were facing problems in doing yoga because of various disabilities. They wanted to practice yoga but couldnt sit on the floor. So we started the concept of chair yoga at our hospital. It is really helpful and their health status has improved. Binda Shukla, 50, is one patient who has been practicing chair yoga for several months. She suffers from upper body stiffness, and has cervical and cholesterol problems. There is definitely relief ever since I started doing meditation and yoga. I feel fresh when I perform deep breathing asanas and am taking less medication now. Even those who have knee joint related complications are comfortable with chair yoga, says Dr Shukla To see whether yoga can improve cognitive functions of patients with severe mental disorder as compared to cardiac controls and to compare improvements in patients with schizophrenia. Time period: Samples of last two years in the study included patients with depression, bipolar disorder, schizophrenia and cardiac controls. Patients were aged between 18-60. Another study was conducted to ascertain randomised control trail on schizophrenia on yoga and non-yoga groups. Exercises: Yoga training included chanting of Om, warm up exercises, breathing (pranayama), various yogic asanas Results: There was an improvement in the speed parameters of most of the cognitive domains among patients with schizophrenia after yoga, compared with the schizophrenia non-yoga group patients. The healthy parameters sustained for six months at least. Tags :Follow YogaFollow International Yoga Day POST A COMMENT READ THIS BJP chief JP Nadda Top BJP leaders take to social media to condemn tweets on farmers' stir by Rihanna, Greta The efficacy of double masking: What health experts have to say Silenced Minority? Airtel, Jio, Vi best prepaid plans with streaming and data benefits under Rs 500 RECOMMENDED WATCH RIGHT NOW 01:00 Watch: Arvind Kejriwal takes first dose of Covid vaccine 02:27 Bengal assembly polls: Mamata Banerjee to file nomination from Nandigram on March 11 00:49 Bengali singer Aditi Munshi joins Trinamool Congress 02:57 Karnataka minister Ramesh Jarkiholi resigns after sex CD scandal 02:01 Good news: Prisoners beautify jail compound in UP's Maharajganj TOP TAKES EC directs petrol pumps to remove hoardings showing PM Modi's photos01:10 EC directs petrol pumps to remove hoardings showing PM Modi's photos BJP calls Rahul Gandhi's push-up challenge violation of code of conduct, writes to EC00:46 BJP calls Rahul Gandhi's push-up challenge violation of code of conduct, writes to EC Delhi Police foils plan to kill 2 Delhi riots accused in Tihar Jail02:30 Delhi Police foils plan to kill 2 Delhi riots accused in Tihar Jail INDIATODAY.IN Download App Andriod AppIOS AppSmartTv App Copyright © 2021 Living Media India Limited. For reprint rights: Syndications Today Covid-19 pandemic: 5 things one should know about heart failure Covid-19 pandemic: 5 things one should know about heart failure KCR's Backward Classes Challenge KCR's Backward Classes Challenge What genes tell us about the risk of developing cancer What genes tell us about the risk of developing cancer Redefining heart health with cutting edge technologies Redefining heart health with cutting edge technologies How yoga can help you to get rid of stress headache How yoga can help you to get rid of stress headache Designer sarees to mark your distinctive impression Designer sarees to mark your distinctive impression Designer sarees to mark your distinctive impression Designer sarees to mark your distinctive impression Do's and dont's to keep in mind while opting for IVF treatment: Here's all you need to know Do's and dont's to keep in mind while opting for IVF treatment: Here's all you need to know Be a style icon with these groovy women handbags Be a style icon with these groovy women handbags India's Avant-Garde India's Avant-Garde"
s = preprocess(s)
s = tfidf.transform([s])
print(mlb.inverse_transform(knnClf.predict(s)))

pickle.dump(knnClf, open('./Data/IAB/IAB_classifier.p','wb'))
pickle.dump(tfidf, open("./Data/IAB/IAB_vectorizer.p", "wb"))
pickle.dump(mlb, open("./Data/IAB/IAB_binarizer.p", "wb"))
loaded_model = pickle.load(open('./Data/IAB/IAB_classifier.p','rb'))
loaded_vectorizer = pickle.load(open('./Data/IAB/IAB_vectorizer.p','rb'))
loaded_binarizer = pickle.load(open('./Data/IAB/IAB_binarizer.p','rb'))