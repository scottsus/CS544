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

def decode(sentence):
    labels = []

    for i, word in enumerate(sentence):
        if i == 0:
            probs = [(label, emission_probs.get((word, label), 0)) for _, label in emission_probs.keys()]
        else:
            probs = [(label, transition_probs.get((labels[-1], label), 0) * emission_probs.get((word, label), 0)) for _, label in transition_probs.keys()]

        best_label = max(probs, key=(lambda prob: prob[1]))[0]
        labels.append(best_label)
    
    return labels

with open('CSCI544_HW2/data/dev.json', 'r') as f:
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

with open('CSCI544_HW2/data/test.json', 'r') as f:
    test_data = json.load(f)

for record in tqdm(test_data):
    sentence = record['sentence']
    labels = decode(sentence)
    record['labels'] = labels

with open('greedy.json', 'w') as f:
    json.dump(test_data, f, indent=4)

print('Successfully wrote to greedy.json')
