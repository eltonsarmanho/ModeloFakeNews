import pandas as pd

from Classificador.LLMCustom import LLMCustom
from Classificador.VerifyContext import VerifyContext

data_path = '../dataset/pre-processed.csv'  # Caminho do seu arquivo CSV

question = "Katia Abreu fez alguma declaração recente sobre política?"

executor = VerifyContext(data_path)
most_similar_index, most_similar_score, most_similar_text = executor.calculate_similarity_in_chunks(question)
is_in_context = most_similar_score >= 0.80

if(is_in_context):
    # Retorna resposta a partir do dataset
    print(most_similar_text);
else:
    #retorna resposta a partir do LLM
    llm = LLMCustom()
    resposta  = llm.pergunta_llm(question)
    print(resposta)