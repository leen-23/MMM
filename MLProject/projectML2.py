
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib
import string
import re
from nltk.corpus import stopwords
from nltk.stem import ISRIStemmer

# Load the CSV file into a DataFrame
data_file_path = "/Users/leenharbi/Desktop/MLdataset.csv"
data = pd.read_csv(data_file_path)

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

# Apply preprocessing to the 'text' column
data['text'] = data['text'].apply(preprocess_text)

# Split data into features and labels
X = data['text']
y = data['Lable']

# Vectorize text data
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Load the trained model and vectorizer
model = joblib.load("/Users/leenharbi/Desktop/PML/MLProject/model.pkl")
vectorizer = joblib.load("/Users/leenharbi/Desktop/PML/MLProject/tfidf_vectorizer.pkl")

# Define the Streamlit app
st.title("Classifying sentiments ")
text_input = st.text_input("Please enter text:")

if text_input:
    # Preprocess the input text
    preprocessed_text = preprocess_text(text_input)
    # Vectorize the preprocessed text
    text_vectorized = vectorizer.transform([preprocessed_text])
    # Predict sentiment using the loaded model
    prediction = model.predict(text_vectorized)
    # Display the sentiment prediction
    if prediction == 1:
        st.write("Positive: إيجابي")
    elif prediction == -1:
        st.write("Negative: سلبي")
    else:
        st.write("Neutral: طبيعي")
