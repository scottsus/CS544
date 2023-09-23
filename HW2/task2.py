import json
from collections import defaultdict

with open('CSCI544_HW2/data/train.json', 'r') as f:
    data = json.load(f)

transmission_counts = defaultdict(int)
emission_counts = defaultdict(int)
initial_labels = defaultdict(int)
label_counts = defaultdict(int)

for record in data:
    sentence, labels = record['sentence'], record['labels']

    first_label = labels[0]
    initial_labels[first_label] += 1

    for i in range(len(labels)):
        word, label = sentence[i], labels[i]
        emission_counts[(word, label)] += 1
        label_counts[label] += 1

        if i < len(labels) - 1:
            next_label = labels[i + 1]
            transmission_counts[(label, next_label)] += 1

emission_probs = {}
for (word, label), emission_count in emission_counts.items():
    emission_prob = emission_count / label_counts[label]
    emission_probs[(word, label)] = emission_prob

transmission_probs = {}
for (label, next_label), transmission_count in transmission_counts.items():
    transmission_prob = transmission_count / label_counts[label]
    transmission_probs[(label, next_label)] = transmission_prob

stringified_emission_probs = {str(key): val for key, val in emission_probs.items()}
stringified_transmission_probs = {str(key): val for key, val in transmission_probs.items()}

model = {
    'transition': stringified_transmission_probs,
    'emission': stringified_emission_probs
}

with open('hmm.json', 'w') as f:
    json.dump(model, f, indent=4)

print(f'Transition parameters: {len(transmission_probs)}')
print(f'Emission parameters: {len(emission_probs)}')

