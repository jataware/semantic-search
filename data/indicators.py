import json
from .corpora import Corpus, CorpusLoader

class Indicators(CorpusLoader):
    @staticmethod
    def get_corpus() -> Corpus[tuple[str,str]]:
        
        with open('data/indicators.jsonl') as f:
            lines = f.readlines()
            indicators = [json.loads(line) for line in lines]

        docs = {}
        for indicator in indicators:
            indicator_id = indicator['_source']['id']
            for out in indicator['_source']['outputs']:
                #name, display name, description, unit, unit description
                description = \
f"""name: {out['name']};
display name: {out['display_name']};
description: {out['description']};
unit: {out['unit']};
unit description: {out['unit_description']};"""
                docs[(indicator_id, out['name'])] = description


        return Corpus(docs)


