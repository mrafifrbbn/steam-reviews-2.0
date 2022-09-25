# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import re
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import joblib
import streamlit as st

# Load the model
model = joblib.load('model/tfidf_sgdc.joblib.dat')

# Function to clean the review
def clean_text(text):
    
    # Remove links
    text = re.sub(r'http\S+', '', text)
    
    # Remove hyperlinks and markups
    text = re.sub("<[a][^>]*>(.+?)</[a]>", 'Link.', text)
    text = re.sub('&gt;', "", text)
    text = re.sub('&#x27;', "'", text)
    text = re.sub('&quot;', '"', text)
    text = re.sub('&#x2F;', ' ', text)
    text = re.sub('<p>', ' ', text)
    text = re.sub('</i>', '', text)
    text = re.sub('&#62;', '', text)
    text = re.sub('<i>', ' ', text)
    text = re.sub("\n", ' ', text)
    
    # Unify whitespaces
    text = re.sub(' +', ' ', text)
    
    # Remove non-alphabetic characters
    text = re.sub(r"[^a-zA-Z?!.,]+", ' ', text)
    
    # Remove punctuation, convert to lowercase
    nopunc = [char.lower() for char in text if char not in punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Remove stop words
    text = nopunc.split()
    no_stopword = [word for word in text if word not in (stopwords.words('english'))]
    
    # Stemming
    snowball_stemmer = SnowballStemmer('english')
    stemmed_word = [snowball_stemmer.stem(word) for word in no_stopword]
    stem = ' '.join(stemmed_word)
    
    # Remove nonsensical words (troll messages) defined as having more than 5 consecutive consonants
    text = re.sub(r'\w*[^aeiouAEIOU\W]{5}\w*', '', stem)
    
    # Check if text is empty after cleaning
    if text=='':
        return np.nan
    else:
        return text


# Function to classify a review using the best model
def sentiment_prediction(input_review):
    
    # Make prediction using the loaded model
    prediction = model.predict([input_review])

    if prediction == 1:
        return "Negative review. The user does not recommend the game."
    else:
        return "Positive review. The user recommends the game."


# Main block of code
def main():
    
    # Title of the app
    st.title("Game Review Sentiment Classification")

    # Get input review from the user
    input_review = st.text_input("User review")
    
    # Clean the review
    input_review = clean_text(input_review)
    
    # Print the results
    sentiment = ''
    if st.button("Sentiment class of the review"):
        sentiment = sentiment_prediction(input_review)
        
    st.success(sentiment)

if __name__ == '__main__':
    main()