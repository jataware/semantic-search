from __future__ import annotations
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

from typing import TypedDict



class NER:
    def __init__(self, model_name="dslim/bert-large-NER"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

    def __call__(self, text: list[str]):
        return self.nlp(text)


class Highlight(TypedDict):
    text: str
    highlight: tuple[str, str] | None


#terminal color/background ANSI codes
ansi_color_codes = {
    'black': 30,
    'red': 31,
    'green': 32,
    'yellow': 33,
    'blue': 34,
    'magenta': 35,
    'cyan': 36,
    'white': 37,
    'bright_black': 90,
    'bright_red': 91,
    'bright_green': 92,
    'bright_yellow': 93,
    'bright_blue': 94,
    'bright_magenta': 95,
    'bright_cyan': 96,
    'bright_white': 97,
}
ansi_background_codes = {
    'black': 40,
    'red': 41,
    'green': 42,
    'yellow': 43,
    'blue': 44,
    'magenta': 45,
    'cyan': 46,
    'white': 47,
    'bright_black': 100,
    'bright_red': 101,
    'bright_green': 102,
    'bright_yellow': 103,
    'bright_blue': 104,
    'bright_magenta': 105,
    'bright_cyan': 106,
    'bright_white': 107,
}

def terminal_highlight_print(highlight_list:list[Highlight]):
    """print the text to the terminal, highlighting at the given spans"""


    # print the chunks
    for span in highlight_list:
        chunk = span['text']
        highlight = span['highlight']
        if highlight is not None:
            # convert the color/background strings to ANSI codes
            color_str, background_str = highlight
            color = ansi_color_codes[color_str]
            background = ansi_background_codes[background_str]
            print(f'\033[{color};{background}m{chunk}\033[0m', end='')
        else:
            print(chunk, end='')
    print()



def spans_to_highlight_list(text: str, spans: list[tuple[int,int,tuple[str,str]]]) -> list[Highlight]:
    """Convert a list of character spans into a list of Highlight objects"""
    spans = sorted(spans, key=lambda x: x[0])
    highlight_list: list[Highlight] = []
    last_end = 0
    for start, end, color in spans:
        if start > last_end:
            highlight_list.append({
                "text": text[last_end:start],
                "highlight": None,

            })
        highlight_list.append({
            "text": text[start:end], 
            "highlight": color
        })
        last_end = end
    
    if last_end < len(text):
        highlight_list.append({
            "text": text[last_end:], 
            "highlight": None,
        })

    return highlight_list






def main():
    from easyrepl import REPL

    model = NER()

    color_map = {
        'B-PER': 'red',
        'I-PER': 'red',
        'B-ORG': 'green',
        'I-ORG': 'green',
        'B-LOC': 'blue',
        'I-LOC': 'blue',
        'B-MISC': 'magenta',
        'I-MISC': 'magenta'
    }

    for query in REPL(history_file='history.txt'):
        results = model(query)
        print(results)
        spans = []
        for result in results:
            spans.append((
                result['start'], 
                result['end'], 
                ('bright_white', color_map[result['entity']])
            ))
        highlight_list = spans_to_highlight_list(query, spans)
        terminal_highlight_print(highlight_list)



if __name__ == "__main__":
    main()