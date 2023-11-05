from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Assume text is a string containing your text data
text = "The capital of france is paris. Most of the people in paris like chicken."

# Split the text into segments (e.g., paragraphs)
segments = text.split('\n')

# Create a TF-IDF vectorizer and fit it to the text segments
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(segments)

def semantic_search(query, threshold=0.2):
    query_vector = vectorizer.transform([query])
    cosine_sim = cosine_similarity(query_vector, tfidf_matrix)
    best_match_index = np.argmax(cosine_sim[0])
    best_match_score = cosine_sim[0][best_match_index]

    if best_match_score >= threshold:
        return segments[best_match_index]
    else:
        return "I'm sorry, I couldn't find a good match for your query."

def chatbot():
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        response = semantic_search(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chatbot()
