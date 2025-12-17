from datasets import load_dataset
import evaluate
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import numpy as np

import torch
print(torch.__version__)

go_emotions_dataset = load_dataset("go_emotions","simplified", split="train")

# number of labels handled by go emotions data set
num_labels = 28

for i in go_emotions_dataset:
    print(i['text'])
    print(i['labels'])
# mapping function can go here if i want to manually add more labels

model_name = "nlpaueb/legal-bert-base-uncased"
tokeniser = BertTokenizer.from_pretrained(model_name, use_safetensors=True)

def tokenise(batch):
    return tokeniser(batch["text"], truncation=True, padding="max_length", max_length=128)

tokenized_train = go_emotions_dataset.map(tokenise, batched=True)

tokenized_test = go_emotions_dataset.map(tokenise, batched=True)

print(tokenized_train)

data_collator = DataCollatorWithPadding(tokenizer=tokeniser)

model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=28,
    problem_type="multi_label_classification",
    use_safetensors=True
)

def compute_metrics(eval_pred):
   load_accuracy = evaluate.load("accuracy")
   load_f1 = evaluate.load("f1")
  
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
   return {"accuracy": accuracy, "f1": f1}

training_args = TrainingArguments(
   output_dir="../custom_bert",
   learning_rate=2e-5,
   per_device_train_batch_size=16,
   per_device_eval_batch_size=16,
   num_train_epochs=2,
   weight_decay=0.01,
   save_strategy="epoch"

)
 
trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_train,
   eval_dataset=tokenized_test,
   tokenizer=tokeniser,
   data_collator=data_collator,
   compute_metrics=compute_metrics,
)

trainer.train()

metrics = trainer.evaluate()
print(metrics)

