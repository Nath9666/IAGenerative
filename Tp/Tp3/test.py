from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# Exemple de corpus pour entraîner le modèle Word2Vec
sentences = [
    ["this", "is", "a", "sample", "sentence"],
    ["word", "to", "vector", "conversion"],
    ["another", "example", "sentence"]
]

# Entraîner le modèle Word2Vec
model = Word2Vec(sentences, vector_size=6, window=5, min_count=1, workers=4)

# Sauvegarder le modèle
model.save("word2vec.model")

# Charger le modèle
model = Word2Vec.load("word2vec.model")

def word_to_vect(word):
    if word in model.wv:
        return model.wv[word]
    else:
        return None

def main():
    word = "sample"
    vector = word_to_vect(word)
    if vector is not None:
        print(f"Vector for '{word}':", vector)
    else:
        print(f"Word '{word}' not in vocabulary")

if __name__ == "__main__":
    main()