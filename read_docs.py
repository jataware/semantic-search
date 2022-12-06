import json
import pdb
from data.corpora import ResearchPapers

#read in the example jsonl documents

def main():
    docs = ResearchPapers.get_corpus()
    pdb.set_trace()

if __name__ == '__main__':
    main()