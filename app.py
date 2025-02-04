import streamlit as st 
import numpy as np 
import pickle 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM Model
model = load_model('next_word_lstm.h5')

# load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
## Funciton to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):] # Ensure the sequence length match max_sequence length
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis =1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word 
    return None

# Streamlit app
st.title("Next word prediction with LSTM and Early Stopping")
input_text = st.text_input("Enter the sequence of words", "To be or not to")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1# Retrive the max sequence length form the input model shaoe
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f'Next Word : {next_word}')