import warnings
warnings.filterwarnings("ignore", message="To copy construct from a tensor*")
from transformers import pipeline

chatbot = pipeline('text-generation', 
model='Felladrin/Llama-68M-Chat-v1', max_new_tokens=256, penalty_alpha=0.6, top_k=4)


system_message = "You are a helpful assistant."
chat = system_message

while True:
    user_message = input("Usuário: ")
    if user_message == "sair":
        break
    print('User:', user_message)
    chat += f"<|im_start|>user\n{user_message}<|im_end|>\n"
    output = chatbot(chat)
    chat = output[0]['generated_text'].split('<|im_start|>assistant\n')[-1].rstrip('<|im_end|>')
    formatted_chat = chat.split('<|im_start|>assistant\n')[-1].rstrip('<|im_end|>')
    print('Assistant:', formatted_chat)

