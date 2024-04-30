# app.py
from flask import Flask, render_template, request
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

#Download the 'stopwords' dataset if not already downloaded
nltk.download('stopwords')

# Load the pre-trained TF-IDF vectorizer
vectorizer = joblib.load("TfidfVectorizer.pkl")

# Load the trained SVC model
model = joblib.load("XGBoost_Model.pkl")

# Initialize the lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

#Function to preprocess and predict the class of an email
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

    # Vectorize the cleaned email
    email_vector = vectorizer.transform([' '.join(cleaned_tokens)])

    # Make the prediction
    predicted_class = model.predict(email_vector)

    return predicted_class[0]

@app.route('/')
def index():
    return render_template('Email_Classif_index.html')

@app.route('/predict', methods=['POST'])
def classify_email():
   
    email_content = request.form.get("email_content")
   
    if email_content:
        predicted_class = predict_email_class(email_content)
        if predicted_class == 1:
            result = "This email is appropriate or Non-Abusive."
        else:
            result = "This email is inappropriate or Abusive."
    else:
        result = "Please enter some email content for classification."
    return result

if __name__ == '__main__':
    app.run(debug=True)
