#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download the 'stopwords' dataset if not already downloaded
nltk.download('stopwords')

# Load the trained SVC model
model = joblib.load("XGBoost_Model.pkl")

# Initialize the lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define and fit the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Function to preprocess and predict the class of an email
def predict_email_class(email_content):
    # Preprocess the email content
    cleaned_email = email_content.lower()
    cleaned_email = re.sub(r'<.*?>!@#$%^&*()', '', cleaned_email)
    cleaned_email = re.sub(r'\d+', '', cleaned_email)
    cleaned_email = cleaned_email.translate(str.maketrans('', '', string.punctuation))
    cleaned_email = cleaned_email.strip()
    pattern = re.compile(r'\s+')
    cleaned_email = pattern.sub(' ', cleaned_email)

    # Tokenize the text and remove stop words
    words = nltk.word_tokenize(cleaned_email)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words]

    # Fit and transform the TF-IDF vectorizer on the cleaned tokens
    email_vector = vectorizer.fit_transform([' '.join(cleaned_tokens)])

    # Make the prediction
    predicted_class = model.predict(email_vector)

    return predicted_class[0]

# Streamlit app UI
st.title("Email Classification")
st.write("Enter an email to check if it's appropriate or not.")

# Text input for the user to enter email content
email_content = st.text_area("Email Content:")

if st.button("Check Email"):
    if email_content:
        predicted_class = predict_email_class(email_content)
        if predicted_class == 1:
            st.write("This email is appropriate or Non-Abusive.")
        else:
            st.write("This email is inappropriate or Abusive.")
    else:
        st.write("Please enter some email content for classification.")


# In[ ]:




