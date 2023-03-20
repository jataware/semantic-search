import torch
from transformers import pipeline
from easyrepl import REPL

def main():
    summarizer = pipeline(
        "summarization",
        "pszemraj/long-t5-tglobal-base-16384-book-summary",
        device=0 if torch.cuda.is_available() else -1,
    )
    for line in REPL():
        with torch.no_grad():
            result = summarizer(line)[0]['summary_text']
        print(f'||| {result}')

if __name__ == "__main__":
    main()
