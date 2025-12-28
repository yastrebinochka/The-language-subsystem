from datasets import load_dataset
import os
import re

def is_russian(text): #проверка на русскоязычные промты необходима, т.к. в датасете есть промты не только на РЯ
    pattern = r'^[а-яА-ЯёЁ\s\-\,\.\!\?\:\;\'"0-9]+$'
    return bool(re.fullmatch(pattern, text))

os.environ["HF_TOKEN"] = "your_token" #токен для доступа в HG
rus_dataset = load_dataset("ZeroAgency/ru-big-russian-dataset", split="train", streaming=True) #открыть датасет, раздел "train" в потоковом режиме
eng_dataset = load_dataset("SoftAge-AI/prompt-eng_dataset", split="train", streaming=True)

rus_raw_text = "rus_raw_prompts.txt" #файлы, куда будут записываться промты
eng_raw_text = "eng_raw_prompts.txt"

with open(rus_raw_text, "w", encoding="utf-8") as f: #создать файл для русскоязычных промтов
    f.write("")

count = 0
with open(rus_raw_text, "a", encoding="utf-8") as f:
    for example in rus_dataset:
        if count >= 100: #ограничитель на 100 промтов
            break
        if example.get('topic') == 'writing': #фильтр на 'topic' == 'writing'
            for message in example['conversation']:
                if message['role'] == 'user':
                    if is_russian(message['content']): #проверка, что текст на русском
                        f.write(f"{count}.{message['content']}\n\n")
                        count += 1

with open(eng_raw_text, "w", encoding="utf-8") as f: #создать файл для англоязычных промтов
    f.write("")

count = 0
with open(eng_raw_text, "a", encoding="utf-8") as f:
    for example in eng_dataset:
        if count >= 100: #ограничитель
            break
        if example.get('Category') == 'Writing': #фильтр на 'Category' == 'Writing'
                f.write(f"{count}.{example.get('Prompt')}\n\n")
                count += 1