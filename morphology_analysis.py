import json
import csv

with open('atomic_p.json', 'r') as f:
    pp = json.load(f)

atomic_p = []    
print(atomic_p[:10])  # Print first 10 atomic prepositions

def get_atomic_p_prop(prop='', counter=5):
# pp is a dict, access preposition as key
    try:
        if prop is not None and prop in ['isAtomicMorph', 'class', 'spellOutHEAD', 'path_p_morphology', 'measure_allowed']:
            for key, value in pp['atomic_p'].items():
                # print(f"key: {key}")
                for el in value:
                    if el == prop:
                        print(f"{key}: {pp['atomic_p'][key][el]} ")
                        counter += 1
                        if counter == 5:
                            break
    except KeyError as e:
        print(f"KeyError: {e} not found in atomic_p")
            
            
    
# print(f"atomic p: {atomic_p[:10]}")  # Print first 10 atomic prepositions
# el = [el for el in pp['atomic_p']]
# print(el)
pp_wordnet_wiki_pop = []
with open('dictionaries/pp_wordnet_wiki_pop.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile, quotechar='|', dialect='excel')
    for row in reader:
        if row['preposition'] == '':
            continue
        pp_wordnet_wiki_pop.append({
            'preposition': row['preposition'],
            'isAtomic': row.get('is_atomic'),
            'isSpatial': row.get('is_spatial')
        })

# print(f"pp populated:{pp_wordnet_wiki_pop[:10]}")  # Print first 10 prepositions from the CSV
# get_atomic_p_prop('isAtomicMorph', counter=10)

# tokenize preposition in pp_wordnet_wiki_pop
unique_tokens = set()
def tokenize_preposition(preposition):
    return preposition.split(' ')

# Example usage
for pp in pp_wordnet_wiki_pop:
    tokens = tokenize_preposition(pp['preposition'])
    # print(f"Tokens for '{pp['preposition']}': {tokens}")    
    unique_tokens.update(tokens)

# print(f"Unique tokens in prepositions: {unique_tokens}")  # Print unique tokens from the prepositions
# print(f"Total unique tokens: {len(unique_tokens)}")  # Print the total number of unique tokens

def decompose_preposition(preposition, unique_tokens, method='substring'):
    result = {}

    if method == 'substring':
        for token in unique_tokens:
            if token in preposition:
                # count = preposition.count(token)
                # remove only one occurrence of token to get the “remainder”
                remainder = preposition.replace(token, "", 1)

                result[token] = {
                    'decomposition': [token, remainder],
                    # 'occurrence': count
                }

    return result

for tokens in unique_tokens:
    for pp in pp_wordnet_wiki_pop:
        print(pp['preposition'])
        print(f"unique_tokens:{tokens}")