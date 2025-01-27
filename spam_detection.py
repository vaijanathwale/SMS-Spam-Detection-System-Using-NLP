import pandas as pd
import numpy as np
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def load_data(file_path):
    """Load the dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path, encoding='latin-1')
        df = df[['v1', 'v2']] 
        df.columns = ['label', 'message']  
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


def preprocess_text(text):
    """Preprocess the input text by removing special characters and converting to lowercase."""
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()  
    return text



def train_model(df):
    """Train a Naive Bayes model on the provided DataFrame."""
    df['message'] = df['message'].apply(preprocess_text)
    
    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)
    
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)
    
    
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
   
    with open('spam_classifier.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    
    return model


def load_model():
    """Load the trained model from a file."""
    with open('spam_classifier.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model


def predict_message(model, message):
    """Predict whether the given message is spam or not."""
    prediction = model.predict([message])
    return prediction[0]


def main():
    """Main function to execute the spam detection system."""
    file_path = r"C:\Users\vaiju\Documents\SMS SPAM DTECTION\spam.csv" 
    df = load_data(file_path)  
    
    try:
        model = load_model()
        print("Loaded existing model.")
    except FileNotFoundError:
        print("No existing model found. Training a new model...")
        model = train_model(df)
    
    user_message = input("Enter the SMS message to check: ").strip()
    if not user_message:
        print("Error: No message entered.")
        return
    
    user_message = preprocess_text(user_message)  # Preprocess the user input
    result = predict_message(model, user_message)

    
    if result == 'spam':
        print("This message is SPAM.")
    else:
        print("This message is NOT SPAM.")

if __name__ == "__main__":
    main()
