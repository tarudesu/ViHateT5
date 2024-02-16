from sklearn.metrics import accuracy_score, f1_score
import os
import argparse
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import unicodedata
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import ast

# Create a parser object
parser = argparse.ArgumentParser(description='Load dataset')
parser.add_argument('--model_id', type=str, help='Repo_ID of fine-tuned model on HuggingFace')
parser.add_argument('--result_filepath', type=str, help='Path for saving the results')


# Parse the arguments
bash_args = parser.parse_args()

# Login to Huggingface
access_token_read = 'YOUR TOKEN HERE'
command = f'huggingface-cli login --token {access_token_read}'
os.system(command)

# Load fine-tuned models from Huggingface
print(f'Loading {bash_args.model_id} from HuggingFace...')

tokenizer = AutoTokenizer.from_pretrained(bash_args.model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(bash_args.model_id)

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
    set_df = set_df[["content", "source", "target", "index_spans"]]

    return set_df

def load_dataset_from_hf(repo_id):

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
vihsd_train_df, vihsd_val_df, vihsd_test_df = load_dataset_from_hf('tarudesu/ViHSD') # You need to upload the dataset to your HuggingFace account and use the repo_id here
victsd_train_df, victsd_val_df, victsd_test_df = load_dataset_from_hf('tarudesu/ViCTSD')
vihos_train_df, vihos_val_df, vihos_test_df = load_dataset_from_hf('tarudesu/ViHOS') # You need to upload the dataset to your HuggingFace account and use the repo_id here

# Inferencing
def generate_output_batch(test_df, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize input texts in the 'source' column
    input_ids_list = [tokenizer.encode(text, return_tensors="pt").to(device) for text in test_df['source']]

    # Initialize tqdm to track progress
    tqdm_input_ids = tqdm(input_ids_list, desc="Generating outputs", unit="text", position=0, leave=True)

    # Generate outputs for each input text
    output_ids_list = [model.generate(input_ids, max_length=256).to(device) for input_ids in tqdm_input_ids]

    # Decode the generated outputs
    output_texts = [tokenizer.decode(output_ids[0], skip_special_tokens=True) for output_ids in output_ids_list]

    # Add the generated outputs to the DataFrame
    test_df['output'] = output_texts

    return test_df


def connect_values(input_list,
                   input_text):  # For [hate]abc[hate] [hate]xyz[hate] in the sample 'abc xyz' => Merge the index and insert the idx of the space between two consecutive words

    input_list = ast.literal_eval(input_list)  # '[]' => []

    modified_list = []
    temp_list = []

    for i in range(len(input_list) - 1):
        modified_list.append(input_list[i])

        if input_list[i + 1] - input_list[i] == 2:
            difference = input_list[i + 1] - input_list[i]
            temp_list.extend(input_list[i] + j for j in range(1, difference))

    # Check each value in temp_list, if input_text[temp_list[i]] is not a space, remove temp_list[i]
    for idx in temp_list:
        if input_text[idx] != " ":
            temp_list.remove(idx)

    modified_list.extend(temp_list)
    modified_list.append(input_list[-1])

    return sorted(modified_list)


def find_and_extract_substrings(original_str, input_str):
    start_tag = '[hate]'
    end_tag = '[hate]'

    input_str = unicodedata.normalize('NFC', input_str.lower())
    original_str = unicodedata.normalize('NFC', original_str.lower())

    # Extract substrings
    substrings = []
    start_index = input_str.find(start_tag)
    while start_index != -1:
        end_index = input_str.find(end_tag, start_index + len(start_tag))
        if end_index != -1:
            substrings.append(input_str[start_index + len(start_tag):end_index])
            start_index = input_str.find(start_tag, end_index + len(end_tag))
        else:
            break

    if not substrings:
        return '[]'

    # Find indices in the original string and merge into one list
    indices_list = []
    for substring in substrings:
        start_index = original_str.find(substring)
        while start_index != -1:
            indices_list.extend(list(range(start_index, start_index + len(substring))))
            start_index = original_str.find(substring, start_index + 1)

    deduplicated_sorted_indices_list = sorted(set(indices_list))

    # deduplicated_sorted_indices_list = connect_values(str(deduplicated_sorted_indices_list), original_str)

    return str(deduplicated_sorted_indices_list)


def process_output_spans(lst_output, text_input):
    processed_lst = []
    for i in range(0, len(lst_output)):
        processed_lst.append(find_and_extract_substrings(text_input[i], lst_output[i]))
    return processed_lst

vihsd_results = generate_output_batch(vihsd_test_df, model, tokenizer)
victsd_results = generate_output_batch(victsd_test_df, model, tokenizer)
vihos_results = generate_output_batch(vihos_test_df, model, tokenizer)
vihos_results['output_spans'] = process_output_spans(vihos_results['output'], vihos_test_df['content'])

# ViHSD: Hate Speech Detection.
# Turn outputs into labels, 'CLEAN', 'OFFENSIVE', 'HATE' to 0, 1, 2
vihsd_results['output'] = vihsd_results['output'].apply(lambda x: 0 if x == 'CLEAN' or x == 'clean' else (1 if x == 'OFFENSIVE' or x == 'offensive' else 2))
vihsd_test_df['target'] = vihsd_test_df['target'].apply(lambda x: 0 if x == 'CLEAN' or x == 'clean' else (1 if x == 'OFFENSIVE' or x == 'offensive' else 2))

# Compute accuracy, weighted f1 score, and macro f1 score
vihsd_accuracy = accuracy_score(vihsd_results['output'], vihsd_test_df['target'])
vihsd_f1_weighted = f1_score(vihsd_results['output'], vihsd_test_df['target'], average='weighted')
vihsd_f1_macro = f1_score(vihsd_results['output'], vihsd_test_df['target'], average='macro')

# ViCTSD: Toxic Speech Detection
# Turn outputs into labels, 'NONE', 'TOXIC' to 0, 1
victsd_results['output'] = victsd_results['output'].apply(lambda x: 0 if x == 'NONE' or x == 'none' else 1)
victsd_test_df['target'] = victsd_test_df['target'].apply(lambda x: 0 if x == 'NONE' or x == 'none' else 1)

# Compute accuracy, weighted f1 score, and macro f1 score
victsd_accuracy = accuracy_score(victsd_results['output'], victsd_test_df['target'])
victsd_f1_weighted = f1_score(victsd_results['output'], victsd_test_df['target'], average='weighted')
victsd_f1_macro = f1_score(victsd_results['output'], victsd_test_df['target'], average='macro')

# ViHOS: Hate Spans Detection
def digitize_spans(vihos_test_df, vihos_results):
    # Convert string representations of lists to actual lists using ast.literal_eval
    vihos_preds = [ast.literal_eval(x) for x in vihos_results['output_spans']]
    vihos_labels = [ast.literal_eval(x) for x in vihos_test_df['index_spans']]

    # Calculate the lengths of the content in vihos_test_df
    vihos_lengths = [len(x) for x in vihos_test_df['content']]

    preds = []
    labels = []

    for i in range(0, len(vihos_lengths)):
        pred = []
        label = []
        for idx in range(0, vihos_lengths[i]):

            if idx in vihos_preds[i]:
                pred.append(1)
            else:
                pred.append(0)

            if idx in vihos_labels[i]:
                label.append(1)
            else:
                label.append(0)

        preds.append(pred)
        labels.append(label)

    return labels, preds

vihos_labels_digits, vihos_preds_digits = digitize_spans(vihos_test_df, vihos_results)

for i in range(0, len(vihos_labels_digits)):
    if len(vihos_labels_digits[i]) != len(vihos_preds_digits[i]):
        print(i)

# Initialize lists for storing accuracy, weighted f1 score, and macro f1 score
vihos_accuracy = []
vihos_f1_weighted = []
vihos_f1_macro = []
#
# Compute accuracy, weighted f1 score, and macro f1 score for vihos_labels_digits and vihos_preds_digits by iterating through each sample in the dataset
for i in range(0, len(vihos_labels_digits)):

    preds_temp = list(vihos_preds_digits[i])
    labels_temp = list(vihos_labels_digits[i])

    vihos_accuracy.append(accuracy_score(labels_temp, preds_temp))
    vihos_f1_weighted.append(f1_score(labels_temp, preds_temp, average='weighted'))
    vihos_f1_macro.append(f1_score(labels_temp, preds_temp, average='macro'))

# Compute the average accuracy, weighted f1 score, and macro f1 score for vihos_labels_digits and vihos_preds_digits by taking the average of the lists
vihos_accuracy = np.mean(vihos_accuracy)
vihos_f1_weighted = np.mean(vihos_f1_weighted)
vihos_f1_macro = np.mean(vihos_f1_macro)

# Print the results in tabular format
print('Results:')
print('ViHSD:')
print(f'Accuracy: {vihsd_accuracy}')
print(f'Weighted F1 Score: {vihsd_f1_weighted}')
print(f'Macro F1 Score: {vihsd_f1_macro}')
print('ViCTSD:')
print(f'Accuracy: {victsd_accuracy}')
print(f'Weighted F1 Score: {victsd_f1_weighted}')
print(f'Macro F1 Score: {victsd_f1_macro}')
print('ViHOS:')
print(f'Accuracy: {vihos_accuracy}')
print(f'Weighted F1 Score: {vihos_f1_weighted}')
print(f'Macro F1 Score: {vihos_f1_macro}')

# Save the results to a CSV file with .4 decimal places

results = {
    'Model': bash_args.model_id,
    'Task': ['ViHSD', 'ViCTSD', 'ViHOS'],
    'Accuracy': [vihsd_accuracy, victsd_accuracy, vihos_accuracy],
    'Weighted F1 Score': [vihsd_f1_weighted, victsd_f1_weighted, vihos_f1_weighted],
    'Macro F1 Score': [vihsd_f1_macro, victsd_f1_macro, vihos_f1_macro]
}

# Round values to 4 decimal places
for key in results:
    if key != 'Model' and key != 'Task':
        results[key] = [round(x, 4) for x in results[key]]

results_df = pd.DataFrame(results)

output_file = '' # Path to the output file

# Append the results to the existing results.csv file
results_df.to_csv(output_file, mode='a', header=False, index=False)

# python t5_evaluate.py --model_id <model_id> --result_filepath <result_filepath>






