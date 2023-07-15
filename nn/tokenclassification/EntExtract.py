from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoTokenizer


base_model = "microsoft/deberta-v3-base"

tokenizer = AutoTokenizer.from_pretrained(base_model)

id2label = {
    0: "O" ,
    1: "B-Product",
    2: "I-Product"
}

label2id = {label : i for i, label in id2label.items() }

from transformers import DataCollatorForTokenClassification
import datasets
from datasets import load_dataset
from transformers import Trainer , TrainingArguments

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


id2label = {
    0: "O" ,
    1: "B-Product",
    2: "I-Product"
}
label2id = {label : i for i, label in id2label.items() }

label_list = list(label2id.keys())


import numpy as np

# https://anaconda.org/conda-forge/seqeval
import evaluate


seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

ds = datasets.Dataset.from_pandas(EntExtractData[EntExtractData.LabelType == "Products"])
shuffled_ds = ds.shuffle(seed=26)


def find_word_index(sentence, char_index):
    # Splitting the sentence into words
    words = sentence.split()
    
    # Finding the word containing the character at char_index
    for i in range(len(words)):
        if char_index < len(words[i]):
            return i
    
        # If char_index is greater than or equal to length of current word,
        # then subtract length of current word from char_index and continue
        char_index -= len(words[i]) + 1
        
    # If no word is found containing the character at char_index
    return -1

def tokenize_and_create_label(example):
    text     = example["ChatOutput"]
    products = example["Labels"].split(product_sep)
    
    # tokenized ids
    tokenized_input = tokenizer(text, truncation=True)
    tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])

    # mapping of token to word index
    word_ids = tokenized_input.word_ids()

    # NER label for tokens
    token_labels = [-100 if word_idx is None else 0 for word_idx in word_ids ]
    if len(example["Labels"]) == 0:
        example.update( tokenized_input )
        # example["tokenized_input"] = tokenized_input
        example["labels"] = token_labels
        # tokenized_input["labels"] = token_labels
        # return tokenized_input
        return example
        
    # need to map each label with words
    # first align with characters
    for product in products:
        # figure out the indices of starting and ending characters of each label
        if product not in text:
            print(f"{product} not in text")
            print(text)
        
        start_index = text.index(product)
        end_index   = start_index + len(product) - 1
        # print(text[start_index:end_index + 1])

        # figure out the indices of words corresponding to start and end characters
        words = text.split()

        start_word_index = find_word_index(text, start_index)
        end_word_index   = find_word_index(text, end_index)

        word_to_token_ids = {}
        for token_id, word_id in enumerate(word_ids):
            if word_id not in word_to_token_ids:
                word_to_token_ids[word_id] = []

            word_to_token_ids[word_id].append(token_id)

        # all tokens within this range is assigned with labels
        start_token_id = word_to_token_ids[start_word_index][0]
        end_token_id = word_to_token_ids[end_word_index][-1]

        token_labels[start_token_id] = label2id["B-Product"]
        for j in range(start_token_id+1, end_token_id):
            # token_labels[start_token_id+1:end_token_id] = label2id["I-Product"]
            token_labels[j] = label2id["I-Product"]
            
    example.update( tokenized_input )
    example["labels"] = token_labels
    
    # tokenized_input["labels"] = token_labels
    
    return example

# tokenize ChatOutput from Bing Chat and assign labels for each token
# Heritage Slim-Fit Oxford Shirt
# B-Product I-Product I-Product I-Product
product_ent_ds = shuffled_ds.map(tokenize_and_create_label, batched=False)

product_ds = product_ent_ds.map(lambda x: x, batched=False, 
                                remove_columns=["ChatOutput", "LabelType", "Labels"]) # "__index_level_0__"

num_train = 7400
train_ds = product_ds.select(range(num_train))
val_ds = product_ds.select(range(num_train, len(product_ds)))

training_args = TrainingArguments(
    output_dir="TrainOut0530",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs = 4,
    weight_decay=0.01,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    warmup_steps  = 100 ,
    logging_steps = 50 ,
    eval_steps    = 50 , 
    save_total_limit = 5 ,
    report_to = "none"
)

model = AutoModelForTokenClassification.from_pretrained(
    base_model , 
    num_labels = len(id2label) , 
    id2label   = id2label, 
    label2id   = label2id
)


trainer = Trainer(
    model = model ,
    args  = training_args ,
    train_dataset   = train_ds ,
    eval_dataset    = val_ds ,
    tokenizer       = tokenizer ,
    data_collator   = data_collator ,
    compute_metrics = compute_metrics , 
)

trainer.train()