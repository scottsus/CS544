import json
from tqdm import tqdm

with open('hmm.json', 'r') as f:
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
all_tags = list({key[0] for key in transition_probs.keys()})

def viterbi_decode(sentence):
    T = len(sentence)
    S = len(all_tags)
    
    # Initialization
    dp = [[0.0 for _ in range(S)] for _ in range(T)]
    backpointer = [[None for _ in range(S)] for _ in range(T)]

    for s, tag in enumerate(all_tags):
        dp[0][s] = emission_probs.get((tag, sentence[0]), 0)

    for t in range(1, T):
        for s, tag in enumerate(all_tags):
            max_prob = -1
            best_prev = None
            for prev_s, prev_tag in enumerate(all_tags):
                prob = dp[t-1][prev_s] * transition_probs.get((prev_tag, tag), 0) * emission_probs.get((tag, sentence[t]), 0)
                if prob > max_prob:
                    max_prob = prob
                    best_prev = prev_s
            dp[t][s] = max_prob
            backpointer[t][s] = best_prev

    # Termination and path recovery
    best_last_tag = dp[-1].index(max(dp[-1]))
    path = [all_tags[best_last_tag]]
    for t in range(T-1, 0, -1):
        best_last_tag = backpointer[t][best_last_tag]
        path.insert(0, all_tags[best_last_tag])

    return path    

with open('CSCI544_HW2/data/dev.json', 'r') as f:
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

with open('CSCI544_HW2/data/test.json', 'r') as f:
    test_data = json.load(f)

for record in tqdm(test_data):
    sentence = record['sentence']
    labels = viterbi_decode(sentence)
    record['labels'] = labels

with open('greedy.json', 'w') as f:
    json.dump(test_data, f, indent=4)

print('Successfully wrote to greedy.json')
