import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import LongformerTokenizer, LongformerModel

def find_product_indices(text, product):
    start_idx = [m.start() for m in re.finditer(re.escape(product), text)]
    end_idx = [idx + len(product) for idx in start_idx]
    return list(zip(start_idx, end_idx))


def get_context_embedding(text, product, tokenizer, model, context_size=5):
    product_indices = find_product_indices(text, product)
    embeddings = []
    for start, end in product_indices:
        start_context = max(0, start - context_size)
        context = text[start_context:end]
        inputs = tokenizer(context, return_tensors="pt", padding=True, truncation=True)
        inputs["attention_mask"] = torch.ones_like(inputs["attention_mask"])
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())
    return embeddings


def get_context_embedding(text, product, tokenizer, model, context_size=5, product_weight=0.8):
    product_indices = find_product_indices(text, product)
    embeddings = []
    for start, end in product_indices:
        start_context = max(0, start - context_size)
        end_context = min(len(text), end + context_size)
        context_before = text[start_context:start]
        context_after = text[end:end_context]
        product_text = text[start:end]

        # Get the embeddings for product
        inputs_product = tokenizer(product_text, return_tensors="pt", padding=True, truncation=True)
        outputs_product = model(**inputs_product)
        product_embedding = outputs_product.last_hidden_state.mean(dim=1).detach().numpy()

        # Get the embeddings for context
        context_text = context_before + " " + context_after
        inputs_context = tokenizer(context_text, return_tensors="pt", padding=True, truncation=True)
        outputs_context = model(**inputs_context)
        context_embedding = outputs_context.last_hidden_state.mean(dim=1).detach().numpy()

        # Combine product and context embeddings using weights
        combined_embedding = product_weight * product_embedding + (1 - product_weight) * context_embedding
        embeddings.append(combined_embedding)
    return embeddings


def products_similarity(text, product1, product2, tokenizer, model):
    embeddings_product1 = get_context_embedding(text, product1, tokenizer, model)
    embeddings_product2 = get_context_embedding(text, product2, tokenizer, model)
    similarities = []
    for emb1 in embeddings_product1:
        for emb2 in embeddings_product2:
            similarity = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))
            similarities.append(similarity[0][0])
    return np.mean(similarities)


def context_similarity(transcript, products):
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
    l = []
    s = []
    text = ''
    for sentence_dict in transcript:
        text += sentence_dict["sentence"]
    for prod1 in products:
        for prod2 in products:
            if prod1 != prod2 and (prod1, prod2) not in l and (prod2, prod1) not in l:
                l.append((prod1, prod2))
                similarity = products_similarity(text, prod1, prod2, tokenizer, model)
                print(prod1, prod2, similarity)
                s.append(similarity)
    return list(zip(l, s))