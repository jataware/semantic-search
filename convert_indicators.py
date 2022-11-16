import json

def main():

    with open('data/indicators.jsonl') as f:
        lines = f.readlines()
        indicators = [json.loads(line) for line in lines]

    descriptions = []
    for indicator in indicators:
        for out in indicator['_source']['outputs']:
            #display name, description, unit, unit description
            description = \
f"""name: {out['name']};
display name: {out['display_name']};
description: {out['description']};
unit: {out['unit']};
unit description: {out['unit_description']};"""
            descriptions.append(description)

    #save the descriptions to a file
    with open('data/descriptions.json', 'w') as f:
        json.dump(descriptions, f)



if __name__ == '__main__':
    main()