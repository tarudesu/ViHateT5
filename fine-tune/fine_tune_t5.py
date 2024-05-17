from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
import argparse
import wandb
import nltk
import os

nltk.download('punkt')

# Set tokenizer parallelism to false
os.environ["TOKENIZERS_PARALLELISM"] = 'False'

access_token_read = '' # Add your HuggingFace access token here
access_token_write = '' # Add your HuggingFace access token here

# Create argument parser
parser = argparse.ArgumentParser(description='Fine-tune T5 model')
parser.add_argument('--save_model_name', type=str, help='Name of fine-tuned model')
parser.add_argument('--pre_trained_ckpt', type=str, help='Path to pre-trained checkpoint on HuggingFace')
parser.add_argument('--output_dir', type=str, help='Path to output directory for saving fine-tuned model')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
parser.add_argument('--wandb_project', type=str, default='ViHateT5', help='Wandb project name')
parser.add_argument('--wandb_run_name', type=str, default='fine-tune', help='Wandb run name')
parser.add_argument('--gpu', type=str, default='0', help='GPU number')

# Parse arguments
bash_args = parser.parse_args()

# Define GPU
os.environ["CUDA_VISIBLE_DEVICES"]=bash_args.gpu


wandb_project = bash_args.wandb_project

# Login to wandb
command = 'wandb login --relogin <ADD WANDB TOKEN HERE>'
os.system(command)

os.environ["WANDB_PROJECT"]=f'{bash_args.wandb_project}'

# Initialize wandb
wandb.init(project=f'{wandb_project}', name=f'{bash_args.wandb_run_name}')


def map_data_vozhsd(set_df):
    map_labels = {
        0: "NONE",
        1: "HATE",
    }

    set_df["source"] = set_df["texts"].apply(lambda x: "hate-speech-detection: " + x)
    set_df["target"] = set_df["labels"].apply(lambda x: map_labels[x])

def map_data_vihsd(set_df):
  map_labels = {
      0: "CLEAN",
      1: "OFFENSIVE",
      2: "HATE",
  }

  set_df["source"] = set_df["free_text"].apply(lambda x: "hate-speech-detection: " + x)
  set_df["target"] = set_df["label_id"].apply(lambda x: map_labels[x])

  set_df = set_df[["source", "target"]]

  return set_df

def map_data_victsd(set_df):
    map_labels = {
        0: "NONE",
        1: "TOXIC",
    }

    set_df["source"] = set_df["Comment"].apply(lambda x: "toxic-speech-detection: " + x)
    set_df["target"] = set_df["Toxicity"].apply(lambda x: map_labels[x])

    set_df = set_df[["source", "target"]]

    return set_df

def process_spans(lst):

    lst = [int(x) for x in lst.strip("[]'").split(', ')]
    result = []
    temp = [lst[0]]

    for i in range(1, len(lst)):
        if lst[i] == lst[i-1] + 1:
            temp.append(lst[i])
        else:
            result.append(temp)
            temp = [lst[i]]

    result.append(temp)
    return result

def add_tags(text, indices):

  if indices == '[]':
    return text

  indices = process_spans(indices)


  for i in range(0, len(indices)):

    # Insert "[HATE]" at index X
    text = text[:indices[i][0]] + "[HATE]" + text[indices[i][0]:]

    # Insert "[\HATE]" at index Y
    text = text[:indices[i][-1]+7] + "[HATE]" + text[indices[i][-1]+7:]

    for j in range(i + 1, len(indices)):
      indices[j] = [x + 12 for x in indices[j]]

  return text

def map_data_vihos(set_df):
    set_df["source"] = set_df["content"].apply(lambda x: "hate-spans-detection: " + x)

    target = []

    for i in range(0, len(set_df['content'])):
        target.append(add_tags(set_df['content'][i], set_df['index_spans'][i]))

    set_df["target"] = target
    set_df = set_df[["source", "target"]]

    return set_df

def load_dataset_from_hf(repo_id):
    # login to huggingface
    command = f'huggingface-cli login --token {access_token_read}'
    os.system(command)

    print(f'Loading {repo_id} from HuggingFace Datasets...')
    data = load_dataset(repo_id, token=True)

    data_train = data['train'].to_pandas().dropna()
    data_val = data['validation'].to_pandas().dropna()
    data_test = data['test'].to_pandas().dropna()

    if repo_id.split('/')[-1] == 'ViHSD':
        data_train = map_data_vihsd(data_train)
        data_val = map_data_vihsd(data_val)
        data_test = map_data_vihsd(data_test)

    elif repo_id.split('/')[-1] == 'ViCTSD':
        data_train = map_data_victsd(data_train)
        data_val = map_data_victsd(data_val)
        data_test = map_data_victsd(data_test)

    elif repo_id.split('/')[-1] == 'ViHOS':
        data_train = map_data_vihos(data_train)
        data_val = map_data_vihos(data_val)
        data_test = map_data_vihos(data_test)

    return data_train, data_val, data_test

print('PROGRESS|Loading datasets...')
vihsd_train_df, vihsd_val_df, vihsd_test_df = load_dataset_from_hf('<your hfID>/ViHSD') # You need to upload the ViHSD dataset to HuggingFace Datasets or modify this code to load from your local machine
victsd_train_df, victsd_val_df, victsd_test_df = load_dataset_from_hf('tarudesu/ViCTSD')
vihos_train_df, vihos_val_df, vihos_test_df = load_dataset_from_hf('your hfID/ViHOS') # You need to upload the ViHOS dataset to HuggingFace Datasets or modify this code to load from your local machine

print('PROGRESS|Concatenating datasets...')
list_of_train_dfs = [vihsd_train_df, victsd_train_df, vihos_train_df]
list_of_val_dfs = [vihsd_val_df, victsd_val_df, vihos_val_df]
list_of_test_dfs = [vihsd_test_df, victsd_test_df, vihos_test_df]

final_train_df = pd.concat(list_of_train_dfs, axis=0)
final_val_df = pd.concat(list_of_val_dfs, axis=0)
final_test_df = pd.concat(list_of_test_dfs, axis=0)

train_data = Dataset.from_pandas(final_train_df)
val_data = Dataset.from_pandas(final_val_df)
test_data = Dataset.from_pandas(final_test_df)

print('PROGRESS|Tokenizing datasets...')
model_id=bash_args.pre_trained_ckpt
tokenizer = AutoTokenizer.from_pretrained(model_id)

if "ViHateT5" in model_id:
    model = T5ForConditionalGeneration.from_pretrained(model_id, from_flax=True) # Because ViHateT5 is a Flax model
else:
    model = T5ForConditionalGeneration.from_pretrained(model_id)

def preprocess_function(sample,padding="max_length"):
    model_inputs = tokenizer(sample["source"], max_length=256, padding=padding, truncation=True)
    labels = tokenizer(sample["target"], max_length=256, padding=padding, truncation=True)
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print('PROGRESS|Tokenizing train, val, test datasets...')
train_tokenized_dataset = train_data.map(preprocess_function, batched=True, remove_columns=train_data.column_names)
val_tokenized_dataset = val_data.map(preprocess_function, batched=True, remove_columns=val_data.column_names)
test_tokenized_dataset = test_data.map(preprocess_function, batched=True, remove_columns=test_data.column_names)
print(f"Keys of tokenized dataset: {list(train_tokenized_dataset.features)}")

# login to huggingface
command = f'huggingface-cli login --token {access_token_write}'
os.system(command)

output_dir=bash_args.output_dir
training_args = Seq2SeqTrainingArguments(
    overwrite_output_dir=True,
    output_dir=output_dir,
    per_device_train_batch_size=bash_args.batch_size,
    per_device_eval_batch_size=bash_args.batch_size,
    learning_rate=bash_args.learning_rate,
    num_train_epochs=bash_args.num_epochs,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="epoch",
    save_strategy="epoch",
    evaluation_strategy="epoch",  # Add this line to set the evaluation strategy
    report_to="wandb",
    run_name=bash_args.wandb_run_name,
    push_to_hub=True,
    load_best_model_at_end=True,
    save_total_limit=1,
    do_train=True,
    do_eval=True,
    predict_with_generate=True,
)

model.config.use_cache = False

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    # Calculate F1 macro
    f1_macro = f1_score(decoded_labels, decoded_preds, average='macro')

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    gen_len_mean = np.mean(prediction_lens)

    return {'f1_macro': round(f1_macro * 100, 4), 'gen_len': round(gen_len_mean, 4)}

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_tokenized_dataset,
    eval_dataset=val_tokenized_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print('PROGRESS|Training...')
trainer.train()

print('PROGRESS|Push to hub...')
trainer.push_to_hub()

wandb.finish()
