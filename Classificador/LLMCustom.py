import pandas as pd
import maritalk

class LLMCustom:

    def __init__(self):
        pass;

    def pergunta_llm(self,pergunta):
        model = maritalk.MariTalk(
            key=minha_chave,
            model="sabia-2-medium"
        )

        messages = [
            {"role": "user", "content": f"{pergunta}"},
        ]

        resposta = model.generate(
            messages,
            do_sample=True,
            max_tokens=200,
            temperature=0.7,
            top_p=0.95)["answer"]

        return resposta

    def update_dataset(df: pd.DataFrame, pergunta, resposta, classe):
        novo_registro = {'Pergunta': pergunta, 'Resposta': resposta, 'LGPD': classe}
        df = df._append(novo_registro, ignore_index=True)
        df.to_csv('../dataset/dataset.csv', index=False)