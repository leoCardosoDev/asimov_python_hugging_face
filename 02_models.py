from transformers import pipeline

models = [
    {
      'name': 'FacebookAI/xlm-roberta-base',
      'token': '<mask>',
    },
    {
      'name': 'neuralmind/bert-large-portuguese-cased',
      'token': "[MASK]",
    },
    {
      'name': 'rufimelo/Legal-BERTimbau-base',
      'token': "[MASK]",
    }
]

for dict_models in models:
    model_name = dict_models['name']
    model_token = dict_models['token']
    print(f"Model: {model_name}")
    model = pipeline('fill-mask', model=model_name)
    frase = f"Esse documento é essencial para {model_token}."
    predicoes = model(frase)
    for predicao in predicoes:
        resposta = predicao['token_str']
        score = predicao['score']
        frase = predicao['sequence']
        score_ajustado = score * 100
        print(f"Resposta: {resposta} | Score: {score_ajustado:.2f}% | Frase: {frase}")