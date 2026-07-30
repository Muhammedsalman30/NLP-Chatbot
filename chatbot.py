import streamlit as st
import json
import random
import joblib

# Load saved model
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Load intents
with open('intents.json') as file:
    data = json.load(file)

def get_response(tag):
    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

st.title("🤖 AI Chatbot")

user_input = st.text_input("You:")

if user_input:
    X_test = vectorizer.transform([user_input])
    
    tag = model.predict(X_test)[0]
    response = get_response(tag)

    st.write("Bot:", response)

if "messages" not in st.session_state:
    st.session_state.messages = []

if user_input:
    st.session_state.messages.append(("You", user_input))
    
    X_test = vectorizer.transform([user_input])
    tag = model.predict(X_test)[0]
    response = get_response(tag)
    
    st.session_state.messages.append(("Bot", response))

for sender, msg in st.session_state.messages:
    st.write(f"**{sender}:** {msg}")