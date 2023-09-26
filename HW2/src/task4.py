import json
import math
from tqdm import tqdm

def load_and_unstringify_data():
    with open('out/hmm.json', 'r') as f:
        model = json.load(f)
    
    stringified_transition_probs = model['transition']
    stringified_emission_probs = model['emission']

    def unstringify(stringified_json):
        unstringified_obj = {}
        for stringified_key, count in stringified_json.items():
            split_keys = stringified_key.strip('()').replace("'", '').split(',')
            unstringified_obj[(split_keys[0].strip(), split_keys[1].strip())] = count
        
        return unstringified_obj

    transition_probs = unstringify(stringified_transition_probs)
    emission_probs = unstringify(stringified_emission_probs)
    labels = list({key[0] for key in transition_probs.keys()})

    return transition_probs, emission_probs, labels

transition_probs, emission_probs, labels = load_and_unstringify_data()
transition_probs = {k: math.log(v) for k, v in transition_probs.items()}
emission_probs = {k: math.log(v) for k, v in emission_probs.items()}

def viterbi_decode(sentence):
    """
    Viterbi Algorithm.

    Idea:

    1. Use a Dynamic Programming table `dp`, where dp[o][s] represents the most likely path so far
       that ends at observation `o` with hidden state `s`.

    2. Also use a backpointer table `bp` where bp[o][s] represents the current path so far with
       observation `o` and hidden state `s` that came directly from observation `o - 1`. In other words,
       retreating to `o - 1` guarantees the most likelihood path that ended in dp[o][s].

    3. Iterating over 1 to observations `O` and over all of states/labels `S`, we try to obtain the
       label index `k` that yields the most likely label that would have led to observation `o` and state `s`.

       This is done by iterating over all other labels and calculating the log sum of the following:
         a) log-probability of the previous most likely observation `o - 1` with the current label `k`
         b) transition probability of the previous label to the current label
         c) emission probability of the current word `sentence[o]` to the current label
       
       This gives us the latest label that enabled us to arrive at `o` and `s` and its associated probability.
    
    4. With the `bp` table populated, the `dp` table's job is done. From the latest label, we iteratively
       backtrack to the previous best label and append it to the start of the `best_path` array simply by
       walking back the `o - 1` portion of `bp` as we mentioned earlier.
    
    """
    O = len(sentence)
    L = len(labels)
    
    dp = [[0 for _ in range(L)] for _ in range(O)]
    bp = [[0 for _ in range(L)] for _ in range(O)]

    for i, label in enumerate(labels):
        dp[0][i] = emission_probs.get((sentence[0], label), float('-9999'))
        bp[0][i] = 0

    for o in range(1, O):
        for s, label in enumerate(labels):
            max_prob, best_prev = float('-inf'), 0
            for k, prev_label in enumerate(labels):
                prob = dp[o - 1][k] + transition_probs.get((prev_label, label), float('-9999')) + emission_probs.get((sentence[o], label), float('-9999'))
                if prob > max_prob:
                    max_prob = prob
                    best_prev = k
            
            dp[o][s] = max_prob
            bp[o][s] = best_prev
    
    best_path = []
    best_label_idx = max(enumerate(dp[-1]), key=lambda probs: probs[1])[0]

    for o in range(O - 1, -1, -1):
        best_path.insert(0, labels[best_label_idx])
        best_label_idx = bp[o][best_label_idx]

    return best_path

with open('starter/data/dev.json', 'r') as f:
    dev_data = json.load(f)

correct, total = 0, 0
for record in tqdm(dev_data):
    sentence, true_labels = record['sentence'], record['labels']
    predicted_labels = viterbi_decode(sentence)

    for i in range(len(true_labels)):
        predicted_label, true_label = predicted_labels[i], true_labels[i]
        if predicted_label == true_label:
            correct += 1

    total += len(true_labels)

accuracy = correct / total
print(f'Accuracy on dev data: {accuracy:.4f}')

with open('starter/data/test.json', 'r') as f:
    test_data = json.load(f)

for record in tqdm(test_data):
    sentence = record['sentence']
    labels = viterbi_decode(sentence)
    record['labels'] = labels

with open('out/viterbi.json', 'w') as f:
    json.dump(test_data, f, indent=4)

print('Successfully wrote to viterbi.json')
