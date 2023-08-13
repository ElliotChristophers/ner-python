import spacy

def spacy_model(transcript):
    nlp = spacy.load("en_core_web_lg")

    entity_types = {}
    entity_entries = {}

    for sentence_dict in transcript:
        sentence = sentence_dict["sentence"]
        doc = nlp(sentence)

        for ent in doc.ents:
            entity_type = ent.label_
            entity_text = ent.text

            if entity_type in entity_types:
                entity_types[entity_type].add(entity_text)
            else:
                entity_types[entity_type] = {entity_text}
            entity_entries[entity_text] = entity_type

    for entity_type, entries in entity_types.items():
        if entity_type == 'PRODUCT':
            break
    return entries


from flair.data import Sentence
from flair.models import SequenceTagger
import logging
logging.getLogger('flair').setLevel(logging.WARNING)

def flair_model(transcript):
    tagger = SequenceTagger.load("flair/ner-english-ontonotes-fast")
    product_entities = []
    for sentence_dict in transcript:
        sentence = sentence_dict["sentence"]
        sentence = Sentence(sentence)
        tagger.predict(sentence)
        for entity in sentence.get_spans('ner'):
            if entity.tag == 'PRODUCT':
                product_entities.append(entity.text)
    return set(product_entities)