import spacy
import json
import re

def process_prompts(input_file, output_file):
    nlp = spacy.load("en_core_web_sm")
    with open(input_file, "r", encoding="utf-8") as file:
        content = file.read()

    matches = re.findall(r'(?:^|\n)(\d+\.\s*)(.*?)(?=\n\d+\.\s*|$)', content, re.DOTALL)
    prompts = [match[1].strip() for match in matches if match[1].strip()] #разбивка по нумерации

    corpus = []
    for idx, prompt_text in enumerate(prompts, start=1): #разметка
        doc = nlp(prompt_text)
        tokens = []
        for token in doc: #морфологическая разметка
            lemma = token.lemma_
            pos = token.pos_

            tokens.append({
                "word": token.text,
                "lemma": lemma,
                "pos": pos,
                "morph": token.morph.to_dict(),
                "intellectual_activity": {}
            })

        root_tokens = [token.text for token in doc if token.dep_ == "ROOT"]
        root = root_tokens[0] if root_tokens else None
        dependencies = []
        for token in doc:
            dependencies.append({
                   "head": token.head.text,
                   "dependent": token.text,
                   "relation": token.dep_
               })

        corpus.append({
            "id": idx,
            "text": prompt_text,
            "tokens": tokens,
            "syntax": {
                "root": root,
                "dependencies": dependencies,
            },
            "language": "en"
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

process_prompts("eng_prompts.txt", "en_corpus.json")