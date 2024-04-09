import numpy as np
import joblib 
from tensorflow.keras.models import load_model  

# Load the model and vectorizer
model = load_model('iq_model.h5')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def predict_iq(input_text):
    input_text_vect = vectorizer.transform([input_text]).toarray()
    predicted_iq = model.predict(input_text_vect)
    return predicted_iq[0][0]

# Example usage
if __name__ == "__main__":
    input_text = input("Enter text for IQ prediction: ")
    predicted_iq = predict_iq(input_text)
    print(f"Predicted IQ Score: {predicted_iq}")

