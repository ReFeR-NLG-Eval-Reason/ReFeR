from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
import time
import json
import os
import random
import argparse
from openai import OpenAI
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
random.seed(42)

gpt_client = OpenAI(api_key=OPENAI_API_KEY)

def generate_answer(answer_context):
    retry = 0
    while True:
        retry += 1
        try:
            completion = gpt_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=answer_context,
                n=1)
            break
        except Exception as e:
            logging.warning(f"retrying due to an error: {e}")
            print(f"retrying due to an error: {e}")
            time.sleep(20)
    return completion

def read_json(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    random.shuffle(data)
    return data[:100]

def write_json(output_path, output_data):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False)

def construct_assistant_message(completion):
    content = completion.choices[0].message.content
    return {"role": "assistant", "content": content}

def process_data_item(data, args, generated_description, intermediate_file_path):
    i = data[0]
    item = data[1]
    if args.task == 'AQuA':
        options = '\n'.join(item['options'])
        question = item['question'] + "Answer Choices are: " + options
        item['Answer'] = item.pop('correct', None)
    elif args.task == "BBH_DU":
        question = item['input']
        item['Answer'] = item.pop('target', None)
    elif args.task == "CSQA":
        options = '\n'.join(item['Options'])
        question = item['Question'] + "Answer Choices are: " + options
    elif args.task == "GSM8k":
        question = item['source']
        item['Answer'] = item['reference'].split("####")[-1]
    elif args.task == "StrategyQA":
        question = item["question"]
        ans = item.pop('answer', None)
        item['Answer'] = ['No', 'Yes'][int(ans)]

    answer = item['Answer']

    # -----------------------------ROUND 0---------------------------------
    if args.task in ['GSM8k']:  # number
        content = """Can you solve the following math problem? {} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response. """.format(question)
    elif args.task in ['AQuA', 'CSQA', 'BBH_DU']:  # option
        content = "Can you answer the following question as accurately as possible? {} Explain your answer, putting the answer in the form (X) at the end of your response.".format(question)
    elif args.task in ['StrategyQA', ]:  # yes or no
        content = "Can you answer the following question as accurately as possible? {} Explain your answer, your answer should be Yes or No at the end of your response.".format(question)
    else:
        raise Exception('failed to construct question, unknown task!')
    agent_contexts = [[{"role": "user", "content": content}] for _ in range(args.agent_num)]
    input_cost = 0
    output_cost = 0
    for agent_context in agent_contexts:
        completion = generate_answer(agent_context)
        input_cost += completion.usage.prompt_tokens
        output_cost += completion.usage.completion_tokens
        assistant_message = construct_assistant_message(completion)
        agent_context.append(assistant_message)
    # -----------------------------ROUND 1---------------------------------
    content = "Review your previous answer and find problems with your answer."
    for _ in range(args.agent_num):
        agent_contexts[_].append({"role": "user", "content": content})

    for agent_context in agent_contexts:
        completion = generate_answer(agent_context)
        input_cost += completion.usage.prompt_tokens
        output_cost += completion.usage.completion_tokens
        assistant_message = construct_assistant_message(completion)
        agent_context.append(assistant_message)
    # -----------------------------ROUND 2---------------------------------
    if args.task in ['GSM8k']:  # number
        content = f"Based on the problems you found, improve your answer. Please reiterate your answer, with your final answer a single numerical number, in the form \\boxed{{answer}}."
    elif args.task in ['AQuA', 'CSQA', 'BBH_DU']:  # option
        content = f"Based on the problems you found, improve your answer. You must choose only one option. Please reiterate your answer, with your final answer a single letter, in the form (X)."
    elif args.task in ['StrategyQA', ]:  # yes or no
        content = f"Based on the problems you found, improve your answer. Please reiterate your answer, your answer should be Yes or No at the end of your response."
    else:
        raise Exception('failed to construct question, unknown task!')
    for _ in range(args.agent_num):
        agent_contexts[_].append({"role": "user", "content": content})

    for agent_context in agent_contexts:
        completion = generate_answer(agent_context)
        input_cost += completion.usage.prompt_tokens
        output_cost += completion.usage.completion_tokens
        assistant_message = construct_assistant_message(completion)
        agent_context.append(assistant_message)
    total_cost = input_cost*0.15 + output_cost*0.6
    processed_data_item = {
        'question': question,
        'answer': answer,
        'agent_contexts': agent_contexts,
        'input_tokens': input_cost,
        'output_tokens': output_cost
    }

    # Save to intermediate file
    with open(intermediate_file_path, 'a', encoding='utf-8') as f:
        json.dump(processed_data_item, f, ensure_ascii=False)

    return processed_data_item, total_cost

def self_correction(args):
    if not args.reload_data:
        generated_description = []
    else:
        # check_dirs_files(dirs=[], files=[args.output_file, ])
        with open(args.output_file, 'r', encoding='utf-8') as f:
            generated_description = json.load(f)
    generated_len = len(generated_description)
    if generated_len:
        logging.info(f'reload from: {args.output_file}')
        logging.info(f'reload data num: {generated_len}')

    all_datas = read_json(args.task_file)
    intermediate_output_file = args.output_file.replace('.json', '_intermediate.json')
    final_cost = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(process_data_item, (i, data), args, generated_description, intermediate_output_file)
            for i, data in enumerate(all_datas) if not (args.reload_data and i < generated_len)
        ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            result, cost = future.result()  # Ensure any exceptions are raised and handled
            generated_description.append(result)
            final_cost += cost

    # Save final results to output file
    write_json(args.output_file, generated_description)
    print(f"Cost of experiment: ${final_cost/1000000}")

def log_param(args):
    args_str = f'\n--------------- single agent parameters ---------------\n'
    for k, v in args.__dict__.items():
        args_str += f'{k} = {v}\n'
    args_str += f'-------------------------------------------------------'
    logging.info(args_str)

if __name__ == "__main__":
    # 1. args
    args_parser = argparse.ArgumentParser(description='peer_review')
    args_parser.add_argument("--exp_folder", type=str, default="Self_Correction",help="Enter the directory name for the experiment results to be stored in")
    args_parser.add_argument('--exp_file', type=str,  default='gpt4o_mini_result.json', help="results file name")
    args_parser.add_argument('--task', type=str, default='GSM8k', choices=['GSM8k','AQuA', 'CSQA', 'BBH_DU'])
    args_parser.add_argument('--agent_num', type=int, default=3) # single agent answer question for 3 times
    args_parser.add_argument('--reload_data', type=bool, default=False, help='Reload existing data to continue from the last checkpoint')
    
    args = args_parser.parse_args()
    log_param(args)

    res_folder = os.path.join("..", "results",args.task, args.exp_folder)
    os.makedirs(res_folder, exist_ok=True)
    save_fp = os.path.join(res_folder,args.exp_file)
    save_fp_intermediate = os.path.join(res_folder, args.exp_file.replace('.json', '_intermediate.json'))
    print(f"For task: {args.task}")
    data_dir = os.path.join('..', 'Datasets',args.task)
    dataset_fp = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if f.endswith('.json')][0]
    args.task_file = dataset_fp
    args.output_file = save_fp

    # 4. self-correction method
    self_correction(args)

    if os.path.exists(save_fp_intermediate):
        os.remove(save_fp_intermediate)
