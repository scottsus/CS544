import json
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

transition_probs, emission_probs, _ = load_and_unstringify_data()

def decode(sentence):
    """
    Normal HMM decoding.

    Idea: Start with the most probable label, then based on that label predict the most likely label
    using the transition and emission probabilities.
    """
    labels = []

    for i, word in enumerate(sentence):
        if i == 0:
            probs = [(label, emission_probs.get((word, label), 0)) for _, label in emission_probs.keys()]
        else:
            probs = [(label, transition_probs.get((labels[-1], label), 0) * emission_probs.get((word, label), 0)) for _, label in transition_probs.keys()]

        best_label = max(probs, key=(lambda prob: prob[1]))[0]
        labels.append(best_label)
    
    return labels

with open('starter/data/dev.json', 'r') as f:
    dev_data = json.load(f)

correct, total = 0, 0
for record in tqdm(dev_data):
    sentence, true_labels = record['sentence'], record['labels']
    predicted_labels = decode(sentence)

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
    labels = decode(sentence)
    record['labels'] = labels

with open('out/greedy.json', 'w') as f:
    json.dump(test_data, f, indent=4)

print('Successfully wrote to greedy.json')
