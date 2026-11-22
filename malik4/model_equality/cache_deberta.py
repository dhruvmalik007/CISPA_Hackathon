from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
AutoModel.from_pretrained("microsoft/deberta-v3-base")
print("cached")
