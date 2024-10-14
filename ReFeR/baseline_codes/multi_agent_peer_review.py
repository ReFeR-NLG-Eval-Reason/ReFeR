import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
import json
import openai
from openai import OpenAI
from tqdm import tqdm
import random
import argparse
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
random.seed(42)

gpt_client = OpenAI(api_key=OPENAI_API_KEY)

def read_json(path:str):
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

def generate_answer(answer_context):
    while True:
        try:
            completion = gpt_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=answer_context,
                n=1)
            break
        except Exception as e:
            logging.warning(f"retrying due to an error: {e}")
            time.sleep(20)
    return completion

def process_data_item(data, args, generated_description, i):
    try:
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
        input_cost = 0
        output_cost = 0
        if args.task in ['GSM8k']:
            content = f"Can you solve the following math problem? {question} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."
        elif args.task in ['AQuA','CSQA', 'BBH_DU']:
            content = f"Can you answer the following question as accurately as possible? {question} Explain your answer, putting the answer in the form (X) at the end of your response."
        elif args.task in ['StrategyQA']:
            content = f"Can you answer the following question as accurately as possible? {question} Explain your answer, your answer should be Yes or No at the end of your response."
        else:
            raise Exception('failed to construct question, unknown task!')

        agent_contexts = [[{"role": "user", "content": content}] for _ in range(args.agent_num)]

        agent_init_ans = None
        agent_feedbacks = [[] for _ in range(args.agent_num)]
        for round_num in range(args.rounds):
            if round_num == 1:
                agent_init_ans = [agent_contexts[k][1]['content'] for k in range(args.agent_num)]

            for j, agent_context in enumerate(agent_contexts):
                if round_num == 0:
                    completion = generate_answer(agent_context)
                    input_cost += completion.usage.prompt_tokens
                    output_cost += completion.usage.completion_tokens
                    assistant_message = construct_assistant_message(completion)
                    agent_context.append(assistant_message)
                elif round_num == 1:
                    ans_to_add = [k for k in range(args.agent_num) if k != j]
                    for index in ans_to_add:
                        init_ans = agent_init_ans[index]
                        content = f"Here is a solution from another agent: \n\n {init_ans}\n\n Please examine this agent's reasoning process step by step and offer feedback on its reasoning. You can rate your confidence in your feedback on a scale from 1-10, where 10 indicates the highest level of confidence."
                        agent_context.append({"role": "user", "content": content})
                        completion = generate_answer(agent_context)
                        input_cost += completion.usage.prompt_tokens
                        output_cost += completion.usage.completion_tokens
                        assistant_message = construct_assistant_message(completion)
                        agent_context.append(assistant_message)
                        agent_feedbacks[index].append(assistant_message)
                elif round_num == 2:
                    agent_feedback = agent_feedbacks[j]
                    agent_num_dict = {1: "one", 2: "two", 3: "three", 4: "four"}
                    content = f"Here are the feedbacks for your solution from the above {agent_num_dict[args.agent_num - 1]} agents:\n\n "
                    for feedback in agent_feedback:
                        content += f"One agent feedback: {feedback['content']} \n\n "

                    if args.task in ['GSM8k']:
                        content += f"Using other agents' solutions and feedbacks as additional information, can you provide your answer to the math problem? \n The original math problem is {question}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."
                    elif args.task in ['AQuA','CSQA', 'BBH_DU']:
                        content += f"Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and other agents' feedback step by step. Put your answer in the form (X) at the end of your response."
                    elif args.task in ['StrategyQA']:
                        content += f"Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and other agents' feedback step by step. Your answer should be Yes or No at the end of your response."
                    else:
                        raise Exception('failed to construct question, unknown task!')

                    agent_context.append({"role": "user", "content": content})
                    completion = generate_answer(agent_context)
                    input_cost += completion.usage.prompt_tokens
                    output_cost += completion.usage.completion_tokens
                    assistant_message = construct_assistant_message(completion)
                    agent_context.append(assistant_message)

        # generated_description.append({
        #     'question': question,
        #     'answer': answer,
        #     'agent_contexts': agent_contexts,
        # })
        total_cost = input_cost*0.15 + output_cost*0.6
        processed_item = {
            'question': question,
            'answer': answer,
            'agent_contexts': agent_contexts,
            'input_tokens': input_cost,
            'output_tokens': output_cost
        }
    except Exception as e:
        logging.error(f"Error processing item {i}: {e}")
    time.sleep(2)
    return processed_item, total_cost

def peer_review(args):
    if not args.reload_data:
        generated_description = []
    else:
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
        futures = [executor.submit(process_data_item, data, args, generated_description, i) 
                   for i, data in enumerate(all_datas) 
                   if not (args.reload_data and i < generated_len)]
        
        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                result, cost = future.result()

                # Write to the intermediate file as each item completes
                with open(intermediate_output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                
                generated_description.append(result)
                final_cost += cost
                pbar.update(1)

    # Write the final output file with all appended results
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(generated_description, f, ensure_ascii=False)
    print(f"Cost of experiment: ${final_cost/1000000}")

def log_param(args):
    args_str = f'\n---------------- peer review parameters ----------------\n'
    for k, v in args.__dict__.items():
        args_str += f'{k} = {v}\n'
    args_str += f'-------------------------------------------------------'
    logging.info(args_str)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser(description='peer_review')
    args_parser.add_argument("--exp_folder", type=str, default="Multi_Agent_Review",help="Enter the directory name for the experiment results to be stored in")
    args_parser.add_argument('--exp_file', type=str,  default='gpt4o_mini_result.json', help="results file name")
    args_parser.add_argument('--task', type=str, default='AQuA', choices=['GSM8k','AQuA', 'CSQA', 'BBH_DU'])

    args_parser.add_argument('--agent_num', type=int, default=3)
    args_parser.add_argument('--rounds', type=int, default=3)
    args_parser.add_argument('--reload_data', type=bool, default=False, help='Reload existing data to continue from the last checkpoint')
    
    args = args_parser.parse_args()

    res_folder = os.path.join("..", "results",args.task, args.exp_folder)
    os.makedirs(res_folder, exist_ok=True)
    save_fp = os.path.join(res_folder,args.exp_file)
    save_fp_intermediate = os.path.join(res_folder, args.exp_file.replace('.json', '_intermediate.json'))
    print(f"For task: {args.task}")
    data_dir = os.path.join('..', 'Datasets',args.task)
    dataset_fp = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if f.endswith('.json')][0]
    args.task_file = dataset_fp
    args.output_file = save_fp


    log_param(args)
    peer_review(args)

    if os.path.exists(save_fp_intermediate):
        os.remove(save_fp_intermediate)
