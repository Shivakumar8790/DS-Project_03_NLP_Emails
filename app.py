from flask import Flask, render_template, request
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline, MarianMTModel, MarianTokenizer
import spacy
from spacy.tokens import Span
from spacy.language import Language

# Initialize the Flask app
app = Flask(__name__)

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Load pre-trained models and vectorizers
vectorizer = joblib.load("TfidfVectorizer.pkl")
model = joblib.load("XGBoost_Model.pkl")

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")
sentiment_analysis = pipeline("sentiment-analysis")
topic_modeling = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Attempt to load the tokenizer and model
try:
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-te")
    translation_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-te")
except Exception as e:
    print(f"Error loading model: {e}")
    tokenizer = None
    translation_model = None

# Preprocess and predict the class of an email
def preprocess_email(email_content):
    email_content = email_content.lower()
    email_content = re.sub(r'<.*?>', '', email_content)
    email_content = re.sub(r'\d+', '', email_content)
    email_content = email_content.translate(str.maketrans('', '', string.punctuation))
    email_content = email_content.strip()
    email_content = re.sub(r'\s+', ' ', email_content)
    words = nltk.word_tokenize(email_content)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words]
    return ' '.join(cleaned_tokens)

def predict_email_class(email_content):
    cleaned_email = preprocess_email(email_content)
    email_vector = vectorizer.transform([cleaned_email])
    predicted_class = model.predict(email_vector)
    return predicted_class[0]

# Perform sentiment analysis
def analyze_sentiment(text):
    result = sentiment_analysis(text)
    return result[0]['label']

# Define custom rules for named entities
@Language.component("custom_entity_ruler")
def custom_entity_ruler(doc):
    custom_entities = []

    for ent in doc.ents:
        # Check if the entity is recognized as an ORG but should be a PERSON
        if ent.label_ == "ORG" and ent.text.lower() in ["shivakumar avunuri"]:
            custom_entities.append(Span(doc, ent.start, ent.end, label="PERSON"))
        # Check if the entity is recognized as a DATE but should be a NUMBER
        elif ent.label_ == "DATE" and ent.text.isdigit() and len(ent.text) == 10:
            custom_entities.append(Span(doc, ent.start, ent.end, label="NUMBER"))
        else:
            custom_entities.append(ent)
    
    doc.ents = custom_entities
    return doc

# Add the custom entity ruler to the SpaCy pipeline
nlp.add_pipe("custom_entity_ruler", after="ner")

# Perform named entity recognition with custom rules
def recognize_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities


# Perform part of speech tagging
def pos_tagging(text):
    doc = nlp(text)
    pos_tags = [(token.text, token.pos_) for token in doc]
    return pos_tags

# Perform topic modeling
def identify_topics(text):
    candidate_labels = ["business", "sports", "politics", "technology", "entertainment", "health"]
    result = topic_modeling(text, candidate_labels)
    return result['labels'][0]

def translate_to_telugu(text):
  """
  Translates the provided text to Telugu using a pre-loaded tokenizer and translation model.

  Args:
      text: The text to be translated (str).

  Returns:
      The translated text in Telugu (str) or an error message if translation fails.
  """
  if tokenizer and translation_model:
    try:
      translated = translation_model.generate(**tokenizer(text, return_tensors="pt", padding=True))
      telugu_text = tokenizer.decode(translated[0], skip_special_tokens=True)
      return telugu_text
    except Exception as e:
      return f"Translation error: {e}"
  else:
    return "Translation model is not available."

@app.route('/')
def index():
    return render_template('Email_Form.html')

@app.route('/predict', methods=['POST'])
def classify_email():
    email_content = request.form.get("email_content")
    if email_content:
        predicted_class = predict_email_class(email_content)
        sentiment = analyze_sentiment(email_content)
        entities = recognize_entities(email_content)
        pos_tags = pos_tagging(email_content)
        topic = identify_topics(email_content)
        translation = translate_to_telugu(email_content)
        
        if predicted_class == 1:
            result = "This email is appropriate or Non-Abusive."
        else:
            result = "This email is inappropriate or Abusive."
        
        return render_template('Result.html', prediction=result, sentiment=sentiment, entities=entities, pos_tags=pos_tags, topic=topic, translation=translation)
    else:
        result = "Please enter some email content for classification."
        return render_template('Result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
