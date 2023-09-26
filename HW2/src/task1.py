import json
from collections import Counter

with open('starter/data/train.json', 'r') as f:
    data = json.load(f)

word_counter = Counter()
for record in data:
    word_counter.update(record['sentence'])

threshold = 3
unk_count = sum(count for word, count in word_counter.items() if count < threshold)
for word in list(word_counter):
    if word_counter[word] < threshold:
        del word_counter[word]
word_counter['<unk>'] = unk_count

sorted_vocab = sorted(word_counter.items(), key=(lambda item: item[1]), reverse=True)
with open('out/vocab.txt', 'w') as f:
    f.write(f'<unk>\t0\t{unk_count}\n')
    
    index = 1
    for word, count in sorted_vocab:
        if word == '<unk>':
            continue

        f.write(f'{word}\t{index}\t{count}\n')
        index += 1

print(f'Threshold: {threshold}')
print(f'Overall size of vocabulary: {len(sorted_vocab)}')
print(f'Final frequency of <unk>: {unk_count}')

