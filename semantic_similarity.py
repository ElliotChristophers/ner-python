from itertools import combinations
import spacy
nlp = spacy.load('en_core_web_lg')

from flair.embeddings import WordEmbeddings
from flair.data import Sentence
from scipy.spatial.distance import cosine
embeddings = WordEmbeddings('glove')

def standalone_similarity(products):
    #use two models -- flair and spacy -- to score the similarity of two products
    #take the arithmetic mean of these as the products' similarity
    #this model only judges the two products' names. no context
    o = []
    p = list(combinations(products, 2))
    for prod1, prod2 in p:
        s_spacy = nlp(prod1).similarity(nlp(prod2))
        sentence1 = Sentence(prod1)
        sentence2 = Sentence(prod2)
        embeddings.embed(sentence1)
        embeddings.embed(sentence2)
        embedding1 = sentence1[0].embedding
        embedding2 = sentence2[0].embedding
        s_flair = 1 - cosine(embedding1, embedding2)
        o.append(((prod1, prod2), (s_spacy+s_flair)*0.5))
    return o
