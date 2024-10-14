import sys
import logging
import time
import json
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from openai import OpenAI
import random

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   

from config import *
random.seed(42)

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the OpenAI client
gpt_client = OpenAI(api_key=OPENAI_API_KEY)


def write_json(output_path, output_data):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False)

def construct_message(agents, question, idx, task):
    if len(agents) == 0:
        return {"role": "user",
                "content": f"Can you double check that your answer is correct? Please reiterate your answer, with your final answer a single numerical number, in the form \\boxed{{answer}}."}

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)
        prefix_string += response

    if task in ['GSM8k']:  # number
        prefix_string += f"\n\n Using the solutions from other agents as additional information, can you provide your answer to the math problem? \n The original math problem is {question}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."
    elif task in ['AQuA', 'CSQA', 'BBH_DU']:  # option
        prefix_string += f"\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that of other agents step by step. Put your answer in the form (X) at the end of your response."
    elif task in ['StrategyQA']:  # yes or no
        prefix_string += f"\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that of other agents step by step. Your answer should be Yes or No at the end of your response."
    else:
        raise Exception('failed to construct question, unknown task!')

    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    content = completion.choices[0].message.content
    return {"role": "assistant", "content": content}


def generate_answer(answer_context):
    while True:
        try:
            completion = gpt_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=answer_context,
                n=1)
            break
        except Exception as e:
            logging.warning(f"Retrying due to an error: {e}")
            time.sleep(20)
    return completion

def read_json(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    random.shuffle(data)
    return data[:100]

def process_single_task(data, args, i, generated_description, intermediate_file_path):
    # Determine question format and answer extraction based on task type
    if args.task == 'AQuA':
        options = '\n'.join(data['options'])
        question = data['question'] + "Answer Choices are: " + options
        data['Answer'] = data.pop('correct', None)
    elif args.task == "BBH_DU":
        question = data['input']
        data['Answer'] = data.pop('target', None)
    elif args.task == "CSQA":
        options = '\n'.join(data['Options'])
        question = data['Question'] + "Answer Choices are: " + options
    elif args.task == "GSM8k":
        question = data['source']
        data['Answer'] = data['reference'].split("####")[-1]
    elif args.task == "StrategyQA":
        question = data["question"]
        ans = data.pop('answer', None)
        data['Answer'] = ['No', 'Yes'][int(ans)]
    answer = data['Answer']

    if args.task in ['GSM8k']:  # number
        content = f"Can you solve the following math problem? {question} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."
    elif args.task in ['AQuA', 'CSQA', 'BBH_DU']:  # option
        content = f"Can you answer the following question as accurately as possible? {question} Explain your answer, putting the answer in the form (X) at the end of your response."
    elif args.task in ['StrategyQA']:  # yes or no
        content = f"Can you answer the following question as accurately as possible? {question} Explain your answer, your answer should be Yes or No at the end of your response."
    else:
        raise Exception('failed to construct question, unknown task!')

    agent_contexts = [[{"role": "user", "content": content}] for _ in range(args.agent_num)]
    input_cost = 0
    output_cost = 0
    # Multi-round debate between agents
    for round in range(args.rounds):
        for j, agent_context in enumerate(agent_contexts):
            if round != 0:
                agent_contexts_other = agent_contexts[: j] + agent_contexts[j + 1:]
                message = construct_message(agent_contexts_other, question, 2 * round - 1, args.task)
                agent_context.append(message)

            completion = generate_answer(agent_context)
            input_cost += completion.usage.prompt_tokens
            output_cost += completion.usage.completion_tokens
            assistant_message = construct_assistant_message(completion)
            agent_context.append(assistant_message)
    total_cost = input_cost*0.15 + output_cost*0.6
    processed_item = {
        'question': question,
        'answer': answer,
        'agent_contexts': agent_contexts,
        'input_tokens': input_cost,
        'output_tokens': output_cost
    }

    with open(intermediate_file_path, 'a', encoding='utf-8') as f:
        json.dump(processed_item, f, ensure_ascii=False)
    
    time.sleep(2)

    return processed_item, total_cost


def debate(args):
    if not args.reload_data:
        generated_description = []
    else:
        with open(args.output_file, 'r', encoding='utf-8') as f:
            generated_description = json.load(f)
    generated_len = len(generated_description)
    if generated_len:
        logging.info(f'Reloaded from: {args.output_file}')
        logging.info(f'Reloaded data num: {generated_len}')

    all_datas = read_json(args.task_file)
    intermediate_output_file = args.output_file.replace('.json', '_intermediate.json')
    final_cost = 0
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_single_task, data, args, i, generated_description, intermediate_output_file)
                   for i, data in enumerate(all_datas)
                   if not (args.reload_data and i < generated_len)]

        # Use tqdm to track progress
        for future in tqdm(as_completed(futures), total=len(futures)):
            result, cost = future.result()  # Ensure any exceptions are raised and handled
            generated_description.append(result)
            final_cost += cost
    
    write_json(args.output_file, generated_description)
    print(f"Cost of experiment: ${final_cost/1000000}")


def log_param(args):
    args_str = f'\n--------------- debate method parameters ---------------\n'
    for k, v in args.__dict__.items():
        args_str += f'{k} = {v}\n'
    args_str += f'-------------------------------------------------------'
    logging.info(args_str)


if __name__ == "__main__":
    # 1. args
    args_parser = argparse.ArgumentParser(description='peer_review')
    args_parser.add_argument("--exp_folder", type=str, default="Multi_Agent_Debate", help="Enter the directory name for the experiment results to be stored in")
    args_parser.add_argument('--exp_file', type=str,  default='gpt4o_mini_result.json', help="Results file name")
    args_parser.add_argument('--task', type=str, default='AQuA', choices=['GSM8k','AQuA', 'CSQA', 'BBH_DU'])
    args_parser.add_argument('--agent_num', type=int, default=3)
    args_parser.add_argument('--rounds', type=int, default=3)
    args_parser.add_argument('--reload_data', type=bool, default=False, help='Reload existing data to continue from the last checkpoint')
    
    args = args_parser.parse_args()

    res_folder = os.path.join("..", "results", args.task, args.exp_folder)
    os.makedirs(res_folder, exist_ok=True)
    save_fp = os.path.join(res_folder, args.exp_file)
    save_fp_intermediate = os.path.join(res_folder, args.exp_file.replace('.json', '_intermediate.json'))

    data_dir = os.path.join('..', 'Datasets', args.task)
    dataset_fp = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')][0]
    args.task_file = dataset_fp
    args.output_file = save_fp
    print(f"Running For Task: {args.task}")

    log_param(args)

    # 4. debate method
    debate(args)

    if os.path.exists(save_fp_intermediate):
        os.remove(save_fp_intermediate)
