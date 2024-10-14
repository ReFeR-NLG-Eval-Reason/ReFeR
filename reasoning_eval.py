import json
from collections import Counter
import re
from prettytable import PrettyTable

def majority_vote(strings):
    if not strings:
        return None  # Handle empty list case
    counter = Counter(strings)
    most_common = counter.most_common(1)
    return most_common[0][0]

def clean_string_gsm8k(input_string):
    return re.sub(r'[^0-9.]+', '', input_string)

def evaluate(task, file_path, keys, b):
    with open(file_path, "r") as f:
        data = json.load(f)

    correct_counts = {key: 0 for key in keys}
    data = data[:100]
    for item in data:
        human = item['Evaluation']["Correct_Ans"].lower().strip().replace(",", "").strip('\n').strip()

        for key in keys:
            if key == 'AC_GPT':
                lst = [x.lower().strip('\n').strip() for x in item['Evaluation'][f"{key}_Ans"]]
                pred = majority_vote(lst)
            elif key == "AC_GPT_1":
                lst = [x.lower().strip('\n').strip() for x in item['Evaluation'][f"AC_GPT_Ans"]]
                pred = lst[b]
            else:
                pred = item['Evaluation'][f"{key}_Ans"].lower().strip('\n').strip()
                
            if task=='GSM8k':
                try:
                    pred = int(float(clean_string_gsm8k(pred.replace("(","").replace(")",""))))
                except Exception:
                    pred = -1
                human = int(human)
            elif task in ["CSQA", "AQuA", "BBH_DU"]:
                pred = pred.split(')')[0].lower()
                human = human.lower()
            
            is_correct = 1 if pred == human else 0
            correct_counts[key] += is_correct

    total_items = len(data)
    accuracies = [round((correct_counts[key] / total_items) * 100, 2) for key in keys]
    return [task] + accuracies

tasks = ["AQuA", "BBH_DU", "CSQA", "GSM8k"]
keys = ['PEER_LLAMA', 'PEER_NEMO', 'PEER_GEMMA', 'AC_GPT', 'AC_GPT_1']

results = {key: [] for key in keys}

for task in tasks:
    file_path = f"results/{task}/ReFeR_Run3/gpt4o_mini_result.json"
    b = 0
    try:
        row = evaluate(task, file_path, keys, b)
        for i, key in enumerate(keys):
            results[key].append(row[i+1])
    except:
        for key in keys:
            results[key].append("N/A")

table = PrettyTable(["Model"] + tasks)
for key in keys:
    # Change the display names for the table
    display_name = {
        'PEER_LLAMA': 'Llama',
        'PEER_NEMO': 'Nemo',
        'PEER_GEMMA': 'Gemma',
        'AC_GPT': 'ReFeR-Turbo',
        'AC_GPT_1': 'ReFeR-Lite'
    }.get(key, key)
    table.add_row([display_name] + results[key])

print(table)