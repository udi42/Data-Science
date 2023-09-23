# Importing the libraries
import re
import unicodedata
import string
import random
import nltk
from nltk.probability import ConditionalFreqDist
import streamlit as st

nltk.download('punkt')
nltk.download('wordnet')

# Filterin the text
def filter_text(text):
    # Normalize text
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # Replace HTML tags with spaces
    text = re.sub('<.*?>', ' ', text)
    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub('\s+', ' ', text)
    # Lowercase the text
    text = text.lower().strip()
    return text

# Cleaning th text
def clean_text(text):
    tokens = nltk.word_tokenize(text)
    wnl = nltk.stem.WordNetLemmatizer()
    cleaned_tokens = [wnl.lemmatize(word) for word in tokens]
    return ' '.join(cleaned_tokens)

def create_ngram_model(text):
    trigrams = list(nltk.ngrams(text.split(), 3, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
    cfdist = ConditionalFreqDist()
    
    for w1, w2, w3 in trigrams:
        cfdist[(w1, w2)][w3] += 1

    # Calculate probabilities
    for w1_w2 in cfdist:
        total_count = float(sum(cfdist[w1_w2].values()))
        for w3 in cfdist[w1_w2]:
            cfdist[w1_w2][w3] /= total_count

    return cfdist

def generate_prediction(model, user_input, num_words=5):
    user_input = clean_text(user_input)
    user_tokens = user_input.split()

    if len(user_tokens) < 2:
        return "Please enter a longer phrase."

    generated_text = list(user_tokens)

    for _ in range(num_words):
        prev_words = tuple(generated_text[-2:])  # Use the last two words for prediction

        if prev_words in model:
            prediction = sorted(model[prev_words], key=lambda w: -model[prev_words][w])
            next_word = random.choices(prediction, k=1)[0]
            generated_text.append(next_word)
        else:
            return "Not enough data to generate a prediction for the given phrase."

    return ' '.join(generated_text)

def main():
    st.title("Text Prediction App")

    with open('nile.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    # Pre-process text
    filtered_text = filter_text(text)
    cleaned_text = clean_text(filtered_text)

    # Make language model
    model = create_ngram_model(cleaned_text)

    user_input = st.text_input("Enter a phrase (or 'q' to quit):")
    
    if user_input.lower() == 'q':
        st.write("Program ended.")
    elif user_input:
        prediction = generate_prediction(model, user_input)
        st.write("Predicted Text:")
        st.write(prediction)

if __name__ == "__main__":
    main()