import streamlit as st
import pickle
import nltk
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stemmer and stopwords
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Spam indicators (must match those used in training)
spam_indicators = [
    'congratulation', 'won', 'prize', 'lottery', 'million', 'dollar', 'rupee', 'click', 
    'link', 'account', 'suspended', 'verify', 'urgent', 'offer', 'free', 'win', 'selected',
    'winner', 'claim', 'immediately', 'limited time', 'expire', 'dear customer', 'dear user',
    'bank', 'payment', 'password', 'login', 'security', 'alert', 'update', 'information',
    'congratulations', 'earn', 'work from home', 'make money', 'investment', 'risk free',
    'guaranteed', 'no experience', 'no skills', 'registration fee', 'one-time payment'
]

def preprocess_text(text):
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords and short words
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    # Stemming
    tokens = [ps.stem(word) for word in tokens]
    
    return ' '.join(tokens)

# Load the model and vectorizer
try:
    with open('enhanced_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('enhanced_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    st.success("Enhanced spam detection model loaded successfully!")
    
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.error("Please make sure you've run the enhanced train_model.py first")
    st.stop()

st.title("Enhanced Email/SMS Spam Classifier 🛡️")

input_sms = st.text_area("Enter the message to check for spam:")

if st.button('Check for Spam'):
    if not input_sms.strip():
        st.warning("Please enter a message to check")
    else:
        try:
            # Preprocess the input
            processed_text = preprocess_text(input_sms)
            
            # Create features
            text_features = vectorizer.transform([processed_text])
            
            # Add spam indicator features
            indicator_features = np.array([
                int(indicator in input_sms.lower()) 
                for indicator in spam_indicators
            ]).reshape(1, -1)
            
            # Combine features
            features = np.hstack((
                text_features.toarray(),
                indicator_features
            ))
            
            # Make prediction
            prediction = model.predict(features)[0]
            proba = model.predict_proba(features)[0]
            
            # Display results
            st.subheader("Analysis Results")
            col1, col2 = st.columns(2)
            
            if prediction == 'spam':
                col1.error(f"🚨 SPAM DETECTED ({proba[1]*100:.1f}% confidence)")
                col2.warning("⚠️ Warning: This message contains characteristics of spam or phishing attempts.")
            else:
                col1.success(f"✅ Not Spam ({proba[0]*100:.1f}% confidence)")
                col2.info("ℹ️ This message appears to be safe.")
            
            # Show detected indicators
            detected_indicators = [
                ind for ind in spam_indicators 
                if ind in input_sms.lower()
            ]
            
            if detected_indicators:
                st.warning("⚠️ Detected spam indicators:")
                for indicator in detected_indicators:
                    st.write(f"- {indicator}")
            
            # Show processed text for debugging
            with st.expander("View processed text"):
                st.code(processed_text)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please try a different message or check the model files")