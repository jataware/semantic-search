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
                description = Indicators.get_indicator_string(out['name'], out['display_name'], out['description'], out['unit'], out['unit_description'])
                docs[(indicator_id, out['name'])] = description


        return Corpus(docs)


    @staticmethod
    def get_indicator_string(name: str, display_name: str, description: str, unit: str, unit_description: str):
        return \
f"""name: {name};
display name: {display_name};
description: {description};
unit: {unit};
unit description: {unit_description};"""