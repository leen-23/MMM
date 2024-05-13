#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing necessary python packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load dataset
olddata = pd.read_csv("MLolddata .csv")
data = pd.read_csv("MLdataset.csv")


# In[2]:


# The old dataset size
print('old dataset size:',olddata.shape)

# Counts the occurrences of each unique label in the old dataset
olddata['Label'].value_counts()


# In[3]:


# Check for duplicate rows
num_duplicates = data.duplicated().sum()
num_duplicates


# In[4]:


# Remove duplicate rows
data.drop_duplicates(inplace=True)


# In[5]:


# Importing necessary  
import string
import re
from nltk.corpus import stopwords
from nltk.stem import ISRIStemmer


# In[6]:


# Arabic stopwords
arabic_stopwords = set(stopwords.words('arabic'))

# Stemmer for Arabic words
stemmer = ISRIStemmer()


def remove_special(text):
    for letter in '#.][!XR':
        text = text.replace(letter, '')
    return text

def remove_punctuations(text):
    arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
    english_punctuations = string.punctuation
    punctuations_list = arabic_punctuations + english_punctuations
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)

def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text

def remove_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)

def clean_str(text):
    search = ["أ", "إ", "آ", "ة", "_", "-", "/", ".", "،", " و ", " يا ", '"', "ـ", "'", "ى", "\", '\n', '\t'", '"', '?', '؟', '!']
    replace = ["ا", "ا", "ا", "ه", " ", " ", "", "", "", " و", " يا", "", "", "", "ي", "", ' ', ' ', ' ', ' ? ', ' ؟ ', ' ! ']
    
    #remove tashkeel
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(p_tashkeel,"", text)
    
    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')

    for i in range(0, len(search)):
        text = text.replace(search[i], replace[i])
    
    #trim    
    text = text.strip()

    return text

def keep_only_arabic(text):
    return re.sub(r'[a-zA-Z?]', '', text).strip()

def preprocess_text(text):
    text = remove_special(text)
    text = remove_punctuations(text)
    text = normalize_arabic(text)
    text = remove_repeating_char(text)
    text = clean_str(text)
    text = keep_only_arabic(text)
    
    tokens = [word for word in text.split() if word not in arabic_stopwords]
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed_tokens)

data['text'] = data['text'].apply(preprocess_text)


# In[7]:


# TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()


# In[8]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Split data into features and labels
X = tfidf_vectorizer.fit_transform(data['text'])
y = data['Lable']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[9]:


from sklearn.naive_bayes import MultinomialNB

# Define the naive bayes model
naive_model = MultinomialNB()

# Fit the model on the training data
naive_model.fit(X_train, y_train)

# Evaluate model performance using default parameter values
print("Performance metrics using default parameters:\n")
print(f"Training Accuracy: {naive_model.score(X_train, y_train):.4f}")
print(f"Testing Accuracy: {naive_model.score(X_test, y_test):.4f}")
print(f"Precision: {precision_score(y_test, naive_model.predict(X_test), average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, naive_model.predict(X_test), average='weighted'):.4f}")
print(f"F1-score: {f1_score(y_test, naive_model.predict(X_test), average='weighted'):.4f}")


# In[10]:


from sklearn.model_selection import GridSearchCV 
from sklearn.preprocessing import RobustScaler
rr_scaler = RobustScaler(with_centering=False)
x_rr =rr_scaler.fit_transform(X)

# Split data into training and testing sets
X_train2, X_test2, y_train2, y_test2 = train_test_split(x_rr, y, test_size=0.2, random_state=42)

param_grid4 = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]} 

grid_search4 = GridSearchCV(MultinomialNB(), param_grid4, cv=5)

grid_search4.fit(X_train2, y_train2)

# print the optimal value of each parameter
print("optimal value of each parameter:",grid_search4.best_params_)
print("Best cross-validation score:",grid_search4.best_score_)


# In[11]:


# Rebuild a model on the training set using the optimum parameters' values
# evaluate the model on the test set
re_naive = grid_search4.best_estimator_
re_naive.fit(X_train2, y_train2)

# Evaluate model performance after tuning the parameters
print("performance after tuning the parameters:\n")
print(f"Training Accuracy: {re_naive.score(X_train2, y_train2):.4f}")
print(f"Testing Accuracy: {re_naive.score(X_test2, y_test2):.4f}")
print(f"Precision: {precision_score(y_test2, re_naive.predict(X_test2), average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test2, re_naive.predict(X_test2), average='weighted'):.4f}")
print(f"F1-score: {f1_score(y_test2, re_naive.predict(X_test2), average='weighted'):.4f}")


# In[12]:


import joblib

joblib.dump(re_naive, 'model.pkl')


# In[13]:


get_ipython().system('pip install streamlit')


# In[14]:




# In[15]:


corpus = data["text"].tolist()


# In[16]:


from sklearn.feature_extraction.text import TfidfVectorizer

# تهيئة وتدريب متجه TF-IDF
vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)

# حفظ المتجه TF-IDF
import joblib
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")


# In[17]:



get_ipython().run_cell_magic('writefile', 'my_streamlit_app.py', 'import streamlit as st\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.svm import SVC\nimport joblib\n\n# تحميل النموذج المدرب والمتجه\nmodel = joblib.load("model.pkl")\nvectorizer = joblib.load("tfidf_vectorizer.pkl")\n\n# تحديد العناصر الواجهة\nst.title("Classifying sentiments ")\ntext_input = st.text_input("Please enter text:")\n\n# التنبؤ بالتصنيف\nif text_input:\n    # تحويل النص إلى متجه TF-IDF\n    text_vectorized = vectorizer.transform([text_input])\n    # التنبؤ باستخدام النموذج\n    prediction = model.predict(text_vectorized)\n    # عرض نتيجة التنبؤ\n    if prediction == 1:\n        st.write("Positive: إيجابي")\n    elif prediction == -1:\n        st.write("Negative: سلبي")\n    else:\n        st.write("Neutral: طبيعي")')


# In[ ]:




