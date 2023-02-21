from data.dart_papers import DartPapers
from easyrepl import REPL
from search.bert_search import BertSentenceSearch
from transformers import BertTokenizerFast, BertModel, logging # type: ignore[import]
import torch
import re


import pdb


stopwords = {
    '[CLS]', '[SEP]',
    # 'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 
}

def good_match(word):
    """check if a word is a stopword, or only contains punctuation"""
    return word not in stopwords and len(re.sub('[^a-zA-Z]', '', word)) > 0


class Highlighter:
    def __init__(self, model='bert-base-uncased'):
        # load BERT tokenizer and model from HuggingFace
        with torch.no_grad():
            logging.set_verbosity_error()
            self.tokenizer = BertTokenizerFast.from_pretrained(model)
            self.model = BertModel.from_pretrained(model)

            # move model to GPU
            if torch.cuda.is_available():
                self.model = self.model.cuda()

        # save the device
        self.device = next(self.model.parameters()).device
    
    def embed(self, s: str) -> torch.Tensor:
        with torch.no_grad():
            tokens = self.tokenizer(s, return_tensors='pt')
            tokens.to(device=self.device)
            embedding = self.model(**tokens).last_hidden_state[0]
            
            return tokens, embedding


def terminal_highlight(text:str, spans:list[tuple[int,int]], background='yellow', color='black'):
    """print the text to the terminal, highlighting at the given spans"""

    #break the text into chunks with a boolean for whether or not to highlight
    chunks = []
    last_end = 0
    for start, end in spans:
        if start > last_end:
            chunks.append((text[last_end:start], False))
        chunks.append((text[start:end], True))
        last_end = end
    if last_end < len(text):
        chunks.append((text[last_end:], False))

    #convert the colors to ANSI codes
    color = {
        'black': 30,
        'red': 31,
        'green': 32,
        'yellow': 33,
        'blue': 34,
        'magenta': 35,
        'cyan': 36,
        'white': 37,
    }[color]
    background = {
        'black': 40,
        'red': 41,
        'green': 42,
        'yellow': 43,
        'blue': 44,
        'magenta': 45,
        'cyan': 46,
        'white': 47,
    }[background]

    # print the chunks
    for chunk, highlight in chunks:
        if highlight:
            print(f'\033[{color};{background}m{chunk}\033[0m', end='')
        else:
            print(chunk, end='')

    print()


 

#blacklist function, reject results with less than some number of alphabetical characters
pattern = re.compile('[^a-zA-Z]')
def get_blacklist(N=500):
    return lambda x: len(x) < N or len(pattern.sub('', x)) < N


def main():
    corpus = DartPapers.get_paragraph_corpus()
    engine = BertSentenceSearch(corpus, save_name=DartPapers.__name__, blacklist=get_blacklist())
    highlighter = Highlighter()

    for query in REPL():
        # perform the search, and collect the top 10 matches
        match_id, score = engine.search(query, n=1)[0]
        raw_text = corpus[match_id]   # get the matching text for the given id

        token_q_obj, embedding_q = highlighter.embed(query)
        token_t_obj, embedding_t = highlighter.embed(raw_text)
        token_t = token_t_obj.tokens()

        matchings = torch.stack([torch.nn.functional.cosine_similarity(q, embedding_t) for q in embedding_q])

        threshold = 0.5

        matched_tokens = (matchings > threshold).any(dim=0)
        matched_indices = [(i,token) for i,(token,match) in enumerate(zip(token_t,matched_tokens)) if match and good_match(token)]
        # print(raw_text)
        # print(matched_indices)
        # print('\n\n\n-----------------------\n\n\n')

        #determine the spans that will be highlighted. This involves combining adjacent tokens from the matched_indices
        #additionally combine together tokens that make up larger words via the "##" prefix (which may be missing from the matched_indices list)
        highlight_token_indices = []
        for i, _ in matched_indices:
            start = i
            end = start
            if token_t[start].startswith('##'):
                while token_t[start-1].startswith('##'):
                    start -= 1
                start -= 1
            while end + 1 < len(token_t) and token_t[end + 1].startswith('##'):
                end += 1
            highlight_token_indices.append((start, end))

        #merge adjacent and overlapping spans into a single span
        highlight_token_indices.sort()
        merged_spans = []
        for span in highlight_token_indices:
            if len(merged_spans) == 0:
                merged_spans.append(span)
            else:
                if span[0] <= merged_spans[-1][1] + 1:
                    merged_spans[-1] = (merged_spans[-1][0], max(merged_spans[-1][1], span[1]))
                else:
                    merged_spans.append(span)

        
        # print(list(enumerate(token_t)))
        # print(merged_spans)

        
        #convert the token indices to character indices in the original text
        highlight_char_spans = []
        for start, end in merged_spans:
            # pdb.set_trace()
            start_char = token_t_obj.token_to_chars(start).start
            end_char = token_t_obj.token_to_chars(end).end
            highlight_char_spans.append((start_char, end_char))

        # print(highlight_char_spans)

        #print out the text, with the matched spans highlighted with a yellow background and black text (using terminal color codes)
        if len(highlight_char_spans) == 0:
            print(raw_text)
        else:
            terminal_highlight(raw_text, highlight_char_spans, 'yellow', 'black')






        # print the results of the search
        # print('Top match:')
        # for match_id, score in matches:
            # raw_text = corpus[match_id]   # get the matching text for the given id
            # print(raw_text, end='\n\n\n') # print the text



    #create a set of word embeddings for the entire corpus
    #have a repl for doing search
    #on search, word embed the query and do cosine similarity with the embeddings from the matched paragraph



if __name__ == '__main__':
    main()