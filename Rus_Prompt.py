from natasha import (
    Doc,
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
)
import json
import re
import pymorphy2

# Загрузка моделей
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)

morph = pymorphy2.MorphAnalyzer() # подключение пайморфи

def process_prompts(input_file, output_file):

    with open(input_file, "r", encoding="utf-8") as file:
        content = file.read()

    prompts = [prompt.strip() for prompt in re.split(r'\d+\.\s*', content)[1:] if prompt.strip()]

    corpus = []
    for idx, prompt_text in enumerate(prompts, start=1):
        doc = Doc(prompt_text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        doc.parse_syntax(syntax_parser)

        tokens_with_morph = []

        for token in doc.tokens:
            # spacy_pos = map_natasha_to_spacy_pos(token.pos)
            # spacy_morph = map_natasha_to_spacy_morph(token.feats)
            parsed = morph.parse(token.text)[0]
            token.lemmatize(morph_vocab) # метод для лемматизации
            tokens_with_morph.append({
                "word": token.text,
                "lemma": token.lemma,
                "pos": token.pos, # сверка с пайморфи parsed.tag.POS if parsed.tag.POS != None else token.pos
                "morph": token.feats,
                "intellectual_activity": {}
            })

        root = None
        for sent in doc.sents:
            for token in sent.tokens:
                if token.rel == "root": # обработка корня предложения
                    root = {
                        "id": int(token.id) - 1,
                        "text": token.text
                    }
                    break

        dependencies = []
        for sent in doc.sents:
            for token in sent.tokens:
                source = int(token.head_id) - 1
                target = int(token.id) - 1

                if hasattr(token, 'rel') and source >= 0 and source < len(sent.tokens) and source != target:
                    relation_info = {
                        "head": sent.tokens[source].text,
                        "dependent": token.text,
                        "relation": token.rel
                    }
                    dependencies.append(relation_info)

        corpus.append({
            "id": idx,
            "text": prompt_text,
            "tokens": tokens_with_morph,
            "syntax": {
                "root": root,
                "dependencies": dependencies,
            },
            "language": "ru"
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)


process_prompts("rus_prompts.txt", "ru_corpus.json")
