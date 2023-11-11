# Results

This MD file substitutes the PDF file containing answers to questions in the assignment.

## Regular BiLSTM

### Validation Set

- Precision: 77.58%
- Recall: 76.79%
- F1: 77.19%

### Test Set

- Precision: 69.64%
- Recall: 67.42%
- F1: 68.51%

## BiLSTM with GloVe Embeddings

### Validation Set

- Precision: 87.24%
- Recall: 88.35%
- F1: 87.79%

### Test Set

- Precision: 81.93%
- Recall: 83.25%
- F1: 82.58%

### GloVe Explanation

GloVe embeddings are pretrained on a large corpus of text and captures a rich semantic relationship between the words, which might help the model associate particular words together. This gives the model a large advantage when processing words in the forward and backwards directions, allowing it to more efficiently capture the semantic relationships between those forward and backwards words.
