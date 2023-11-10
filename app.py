import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import string
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from spellchecker import SpellChecker

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

# Create a spell-checker instance
spell = SpellChecker()

# Function to split text into sentences and preprocess them
def preprocess_and_split_into_sentences(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences

# Preprocessing function to clean, lemmatize, and correct spelling in the text
def preprocess_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])

    words = text.split()
    corrected_words = [spell.correction(word) if spell.correction(word) else word for word in words]

    return " ".join([token.lemma_ for token in nlp(" ".join(corrected_words))])

# Load text from a txt file
def load_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Define a function to add positional encoding to embeddings
def add_positional_encoding(embeddings):
    seq_len, emb_dim = embeddings.shape
    position = np.arange(0, seq_len, dtype=np.float32)
    div_term = np.exp(np.arange(0, emb_dim, 2) * -(np.log(10000.0) / emb_dim))
    positions = np.outer(position, div_term)
    positional_encodings = np.zeros((seq_len, emb_dim), dtype=np.float32)
    positional_encodings[:, 0::2] = np.sin(positions)
    positional_encodings[:, 1::2] = np.cos(positions)
    return embeddings + positional_encodings

# Train Word2Vec model with positional encoding
def train_word2vec_model(text):
    tokenized_text = [preprocess_text(sentence).split() for sentence in text]
    model = Word2Vec(tokenized_text, vector_size=100, window=5, min_count=1, sg=0)

    # Add positional encoding to the word embeddings
    word_embeddings = model.wv.vectors
    word_embeddings_with_position = add_positional_encoding(word_embeddings)

    # Update the model's word vectors with positional encoding
    model.wv.vectors = word_embeddings_with_position

    return model

# Modify this path to point to your txt file
file_path = 'z.txt'  # Replace with the path to your text file
text = load_text_from_file(file_path)

segments = preprocess_and_split_into_sentences(text)

# Train Word2Vec model with positional encoding
word2vec_model = train_word2vec_model(segments)

# Create a TF-IDF vectorizer with bi-grams and preprocess the text
vectorizer = TfidfVectorizer(ngram_range=(1, 2), preprocessor=preprocess_text)
tfidf_matrix = vectorizer.fit_transform(segments)

def semantic_search(query, threshold=0.2):
    query = preprocess_text(query)
    query_tokens = query.split()

    query_vector = np.mean([word2vec_model.wv[word] for word in query_tokens if word in word2vec_model.wv], axis=0)

    sentence_vectors = []
    for sentence in segments:
        sentence_vector = np.mean([word2vec_model.wv[word] for word in preprocess_text(sentence).split() if word in word2vec_model.wv], axis=0)
        sentence_vectors.append(sentence_vector)

    cosine_sim = [cosine_similarity([query_vector], [sentence_vector])[0][0] for sentence_vector in sentence_vectors]

    good_matches = [(segments[i], score) for i, score in enumerate(cosine_sim) if score >= threshold]
    good_matches.sort(key=lambda x: x[1], reverse=True)
    return good_matches

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
        responses = semantic_search(user_input)
        if responses:
            best_response, _ = responses[0]
            print(f"Chatbot: {best_response}")
        else:
            print("Chatbot: I couldn't find a good match for your query. Please try rephrasing.")

if __name__ == "__main__":
    chatbot()
