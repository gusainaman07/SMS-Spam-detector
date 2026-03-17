import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import re
import pickle

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Enhanced preprocessing function
ps = PorterStemmer()

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
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    # Stemming
    tokens = [ps.stem(word) for word in tokens]
    
    return ' '.join(tokens)

# Load and prepare data
df = pd.read_csv('spam.csv', encoding='latin1')
df = df[['v1', 'v2']].rename(columns={'v1': 'target', 'v2': 'text'})

# Add some common spam indicators
spam_indicators = [
    'congratulation', 'won', 'prize', 'lottery', 'million', 'dollar', 'rupee', 'click', 
    'link', 'account', 'suspended', 'verify', 'urgent', 'offer', 'free', 'win', 'selected',
    'winner', 'claim', 'immediately', 'limited time', 'expire', 'dear customer', 'dear user',
    'bank', 'payment', 'password', 'login', 'security', 'alert', 'update', 'information',
    'congratulations', 'earn', 'work from home', 'make money', 'investment', 'risk free',
    'guaranteed', 'no experience', 'no skills', 'registration fee', 'one-time payment'
]

# Add synthetic spam examples
synthetic_spam = [
    "Congratulations! You've won $1,000,000! Click here to claim your prize!",
    "URGENT: Your bank account has been compromised. Verify your details now!",
    "Earn $500 daily working from home. No experience needed!",
    "Your account will be suspended. Click here to verify: http://fakebank.com/verify",
    "You've been selected for a special offer! Limited time only!",
    "Claim your iPhone 13 for free! Just pay shipping and handling.",
    "Your subscription will renew automatically. Click here to cancel.",
    "Security Alert: Unusual login attempt. Verify your account now!",
    "You've been pre-approved for a loan with 0% interest!",
    "Your package delivery is pending. Click to schedule delivery."
]

# Add synthetic data to the dataset
synthetic_df = pd.DataFrame({
    'target': ['spam'] * len(synthetic_spam),
    'text': synthetic_spam
})

df = pd.concat([df, synthetic_df], ignore_index=True)

# Apply preprocessing
df['processed_text'] = df['text'].apply(preprocess_text)

# Add spam indicator features
for indicator in spam_indicators:
    df[indicator] = df['text'].str.lower().str.contains(indicator, regex=False).astype(int)

# Split the data
X = df[['processed_text'] + spam_indicators]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create and fit the vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),  # Consider unigrams and bigrams
    min_df=2,
    max_df=0.8,
    stop_words='english'
)

# Transform text features
X_train_text = vectorizer.fit_transform(X_train['processed_text'])
X_test_text = vectorizer.transform(X_test['processed_text'])

# Combine text features with indicator features
X_train_final = np.hstack((
    X_train_text.toarray(),
    X_train[spam_indicators].values
))

X_test_final = np.hstack((
    X_test_text.toarray(),
    X_test[spam_indicators].values
))

# Train the model with class weights (to handle imbalanced data)
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    C=0.5,
    penalty='l2'
)
model.fit(X_train_final, y_train)

# Evaluate the model
y_pred = model.predict(X_test_final)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
with open('enhanced_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('enhanced_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("\nModel and vectorizer saved successfully!")
print("Spam indicators used:", spam_indicators)