import streamlit as st
import joblib

import pandas as pd
import re
from urllib.parse import urlparse
from collections import Counter
import numpy as np

## Note it is better to have this feature extraction function in a seperate file and import it in the feature extraction notebook and in the app script. However that has been ignored for now.

tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

common_short_url = ['bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly', 'buff.ly', 'is.gd', 'adf.ly', 'sniply.io', 'dub.co', 'short.io', 'ZipZy.in']

def calculate_entropy(s):
    probs = [freq / len(s) for freq in Counter(s).values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)

def preprocess_url(url):
    """Extract words from URL (remove special characters and split by delimiters)."""
    url = re.sub(r"https?://", "", url)  
    url = re.sub(r"[^a-zA-Z0-9]", " ", url)  
    return url.lower()

def extract_url_features(url):
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path
    
    unsafe_chars = set(' "<>#%{}|\\^[]~`')
    suspicious_words = {"login", "secure", "verify", "bank", "auth", "account", "update", "confirm", "signin", "wp-login", "validate", "submit"}
    return {
        "url_len": len(url),
        "num_unsafe_chars": sum(1 for char in url if char in unsafe_chars),
        "num_digits": sum(c.isdigit() for c in url),
        "num_subdomains": domain.count('.'),
        "is_ip": bool(re.match(r'\d+\.\d+\.\d+\.\d+', domain)), 
        "num_params": url.count('?'),
        "num_slashes": url.count('/'),    #directory depth and obfuscation
        "contains_suspicious_keywords": any(word in url.lower() for word in suspicious_words),
        "contains_suspicious_file_extension": any(ext in path.lower() for ext in ['.exe', '.zip', '.js', '.php']),
        "short_url": any(short in domain for short in common_short_url),  
        "url_entropy": calculate_entropy(url),
        "has_https": url.startswith("https")
    }


model = joblib.load('models/LightGBM_model.pkl')
st.title("🔍 Malicious URL Detector")
url_input = st.text_input("Enter a URL to analyze:")

if st.button("Check URL"):
    if url_input.strip() == "":
        st.warning("Please enter a valid URL.")
    else:
        features = extract_url_features(url_input)
        features_df = pd.DataFrame([features])

        #calculate tf-idf
        processed_url = preprocess_url(url_input)
        tfidf_vector = tfidf_vectorizer.transform([processed_url]).toarray()
        tfidf_score = np.sum(tfidf_vector)  

        features_df["tfidf_score"] = tfidf_score   
        prediction = model.predict(features_df)
        st.success(f"🔎 Prediction: **{prediction}**")
