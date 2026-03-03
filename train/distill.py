import os.path

from model2vec.distill import distill

BASE_MODELS = [
    "BAAI/bge-base-en-v1.5",
    "sentence-transformers/LaBSE",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-small",
    "google-bert/bert-base-multilingual-cased",
    
    "BSC-LT/MrBERT",
    "BSC-LT/MrBERT-es", # es / en
    "BSC-LT/MrBERT-ca", # ca/ en
    "BSC-LT/MrBERT-legal", # es / en
    "BSC-LT/MrBERT-biomed", # es / en
    "BSC-LT/MrBERT-science", # es / en

    "dvilares/bertinho-gl-small-cased", # gl

    "renneruan/BERTomelo-ModernBERT-Base-1k",
    "Le0ssa/bertha-portuguese-small",  # pt-BR
    "neuralmind/bert-base-portuguese-cased",  # pt-BR
    "neuralmind/bert-large-portuguese-cased", # pt-BR

    "projecte-aina/roberta-base-ca-v2",  # ca
    "projecte-aina/roberta-large-ca-v2",  # ca
    "projecte-aina/distilroberta-base-ca-v2", # ca

    "HiTZ/BERnaT-base",  # eu
    "HiTZ/BERnaT-medium", # eu
    "HiTZ/BERnaT-large", # eu
    "HiTZ/EriBERTa-base"
]

export_path = os.path.join(os.path.dirname(__file__), "distilled")
os.makedirs(export_path, exist_ok=True)

for m in BASE_MODELS:
    path = os.path.join(export_path,  m.split("/")[-1])
    if os.path.exists(path):
        continue

    # Distill a Sentence Transformer model
    m2v_model = distill(model_name=m)

    # Save the model
    m2v_model.save_pretrained(path)