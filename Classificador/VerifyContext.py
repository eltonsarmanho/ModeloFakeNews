import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class VerifyContext:
    def __init__(self,data_path):
        # Carregar o CSV com perguntas rotuladas
        df = pd.read_csv(data_path)
        self.text = df['preprocessed_news'].tolist()





    def calculate_similarity_in_chunks(self, input_text, chunk_size=500):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        texts = self.text
        vectorizer = TfidfVectorizer()
        input_vector = vectorizer.fit_transform([input_text]).toarray()

        max_similarity = 0
        most_similar_text = ""
        most_similar_index = -1

        for i in range(0, len(texts), chunk_size):
            chunk_texts = texts[i:i + chunk_size]
            chunk_vector = vectorizer.transform(chunk_texts).toarray()
            similarities = cosine_similarity(input_vector, chunk_vector).flatten()

            if similarities.max() > max_similarity:
                max_similarity = similarities.max()
                most_similar_index = i + similarities.argmax()
                most_similar_text = texts[most_similar_index]

        return most_similar_index, max_similarity, most_similar_text

