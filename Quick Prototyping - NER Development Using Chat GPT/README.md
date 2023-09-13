# Quick Prototyping: NER Development using ChatGPT
The notebook in this folder leverages the power of ChatGPT to quickly prototype an NER model. The NER model leverages a pretrained model, dbmdz/bert-large-cased-finetuned-conll03-english, and an open source dataset, conll2003.

## Prompting ChatGPT
1. _Write a script to train a named entity recognition model using the conll2003 dataset  from the datasets_
### Output
```
import datasets
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer

# Load the CoNLL-2003 dataset
dataset = datasets.load_dataset("conll2003")

# Load a pre-trained NER model and tokenizer
model_checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

# Define the data processing function
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] - 1)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Tokenize and preprocess the dataset
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# Define training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    output_dir="./ner_model",
    evaluation_strategy="steps",
    eval_steps=500,
    save_total_limit=2,
    save_steps=500,
    learning_rate=2e-5,
    num_train_epochs=3,
    report_to="tensorboard",
    load_best_model_at_end=True,
)

# Define Trainer and start training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

trainer.train()

# Save the model
model.save_pretrained("./ner_model")

# Evaluate the model
results = trainer.evaluate()
print(results)

# Optionally, you can fine-tune the model further or use it for NER tasks.

```
However, this yieled the error:
```
ValueError: expected sequence of length 30 at dim 1 (got 16)
```
2. I informed ChatGPT of this error which yielded corrected code. The tokenization function was modified to pad the input sequences to a fixed length.
```
# Ensure all examples have the same sequence length (e.g., 30 tokens)
# You can set this consistent sequence length as a hyperparameter.
consistent_sequence_length = 30

# Tokenize and pad the data to the consistent sequence length
def tokenize_and_pad(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",  # Pad to the specified length
        max_length=consistent_sequence_length,
        is_split_into_words=True,
    )

    # Align and pad the label data as well
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        label_ids = label[:consistent_sequence_length] + [0] * (consistent_sequence_length - len(label))
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Tokenize and preprocess the dataset
tokenized_datasets = dataset.map(tokenize_and_pad, batched=True)
```

**This yielded successfull training:**

![image](https://github.com/justinpontalba/Projects/assets/58340716/9d382602-b0d5-4b81-a00c-a3e9a0145c93)
