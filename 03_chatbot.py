from transformers import pipeline

chatbot = pipeline('text-generation', model='Felladrin/Llama-68M-Chat-v1', max_new_tokens=256)
pergunta = "Hi, what is your name?"
resposta = chatbot(pergunta)
print(resposta)
