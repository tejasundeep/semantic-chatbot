from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy
import string

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

# Example text
text = "The capital of France is Paris. Most of the people in Paris like chicken."

# Function to split text into sentences and preprocess them
def preprocess_and_split_into_sentences(text):
    # Using spacy for better sentence segmentation
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences

# Preprocessing function to clean and lemmatize the text
def preprocess_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    return " ".join([token.lemma_ for token in nlp(text)])

segments = preprocess_and_split_into_sentences(text)

# Create a TF-IDF vectorizer with bi-grams and preprocess the text
vectorizer = TfidfVectorizer(ngram_range=(1,2), preprocessor=preprocess_text)
tfidf_matrix = vectorizer.fit_transform(segments)

def semantic_search(query, threshold=0.2):  # Increased threshold
    query = preprocess_text(query)  # Preprocess query
    query_vector = vectorizer.transform([query])
    cosine_sim = cosine_similarity(query_vector, tfidf_matrix)
    
    good_matches = [segments[i] for i, score in enumerate(cosine_sim[0]) if score >= threshold]
    joined_matches = ' '.join(good_matches) if good_matches else "I couldn't find a good match for your query. Please try rephrasing."
    
    return joined_matches

def chatbot():
    print("Chatbot activated. Type 'quit' or 'exit' to stop.")
    while True:
        user_input = input("You: ")
        if not user_input.strip():
            print("Chatbot: Please enter some text to search.")
            continue
        if user_input.lower() in ["quit", "exit"]:
            print("Chatbot: Goodbye!")
            break
        response = semantic_search(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chatbot()
