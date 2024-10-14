import json
import argparse
import tqdm
import time
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from openai import OpenAI
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import google.generativeai as genai #pip install -q -U google-generativeai
from together import Together
# from reasoning_eval import evaluate
from config import *
# suppress warnings
import logging
logging.getLogger().setLevel(logging.ERROR)


def process_prompts(task, instance, peer_prompt_temp, AC_prompt_temp):

    if task == 'AQuA':
        options = '\n'.join(instance['options'])
        peer_prompt = peer_prompt_temp.replace('{{question}}', instance['question']).replace('{{options}}',options)
        AC_prompt = AC_prompt_temp.replace('{{question}}', instance['question']).replace('{{options}}',options)
    elif task == "BBH_DU":
        ### Options are also given in question itself for bbh-du
        peer_prompt = peer_prompt_temp.replace('{{question}}', instance['input'])
        AC_prompt = AC_prompt_temp.replace('{{question}}', instance['input'])
    elif task == "CSQA":
        options = '\n'.join(instance['Options'])
        peer_prompt = peer_prompt_temp.replace('{{question}}', instance['Question']).replace('{{options}}',options)
        AC_prompt = AC_prompt_temp.replace('{{question}}', instance['Question']).replace('{{options}}',options)
    elif task == "GSM8k":
        #No options in GSM8k
        peer_prompt = peer_prompt_temp.replace('{{question}}', instance['source'])
        AC_prompt = AC_prompt_temp.replace('{{question}}', instance['source'])

    return peer_prompt,AC_prompt    

def ReFeR(instance, task,peer_prompt_temp,AC_prompt_temp,save_fp):

    peer_prompt, AC_prompt = process_prompts(task,instance,peer_prompt_temp,AC_prompt_temp)
    instance['peer_prompt'] = peer_prompt
    if task == 'GSM8k':
        instance['Answer'] = instance['reference'].split("####")[-1]
    elif task == "AQuA":
        instance['Answer'] = instance.pop('correct', None)
    elif task == "BBH_DU":
        instance['Answer'] = instance.pop('target', None)

    retry = 0
    while True:
        if retry>4:
            break
        # llama3.1-8B
        try:
            _response = togetherai_client.chat.completions.create(messages=[
                            {
                                "role": "user",
                                "content": peer_prompt,
                            }], model=args.llama_model,
                            temperature=1,
                            top_p=1,
                            max_tokens=256
                            )

            all_responses_1 = [_response.choices[i].message.content for i in
                                range(len(_response.choices))]
            all_responses_1 = [x.replace("*", "") for x in all_responses_1]
            instance['PEER_LLAMA'] = all_responses_1
            instance['PEER_LLAMA_input_tokens'] = _response.usage.prompt_tokens
            instance['PEER_LLAMA_output_tokens'] = _response.usage.completion_tokens
        except Exception as e:
            with open("errors.txt", "a") as f:
                f.write("LLAMA: " + str(e) + "\n")
            if ("limit" in str(e)):
                time.sleep(3)
            else:
                break
        # Mistral-Nemo 12B
        try:
            messages = [ChatMessage(role="user", content=peer_prompt)]
            _response = mist_client.chat(
                model=args.nemo_model,
                messages=messages,
                temperature=1,
                top_p=1,
                max_tokens=256
            )
            all_responses_2 = [_response.choices[i].message.content for i in
                            range(len(_response.choices))]      
            instance['PEER_NEMO'] = all_responses_2
            instance['PEER_NEMO_input_tokens'] = _response.usage.prompt_tokens
            instance['PEER_NEMO_output_tokens'] = _response.usage.completion_tokens
        except Exception as e:
            with open("errors.txt", "a") as f:
                f.write("NEMO: " + str(e) + "\n")
            if ("limit" in str(e)):
                time.sleep(3)
            else:
                break
        # GEMMA2-9B
        try:
            _response = togetherai_client.chat.completions.create(messages=[
                            {
                                "role": "user",
                                "content": peer_prompt,
                            }], model=args.gemma_model,
                            temperature=1,
                            top_p=1,
                            max_tokens=256
                            )

            all_responses_3 = [_response.choices[i].message.content for i in
                                range(len(_response.choices))]
            # all_responses_3 = [x.replace("*", "") for x in all_responses_3]
            instance['PEER_GEMMA'] = all_responses_3
            instance['PEER_GEMMA_input_tokens'] = _response.usage.prompt_tokens
            instance['PEER_GEMMA_output_tokens'] = _response.usage.completion_tokens

        except Exception as e:
            with open("errors.txt", "a") as f:
                f.write("GEMMA: " + str(e) + "\n")
            if ("limit" in str(e)):
                time.sleep(3)
            else:
                break
        
        #get answers using regex Answer
        
        try:
            instance["Evaluation"] = {}

            if task in ['GSM8k']:
                

                instance["Evaluation"]['Correct_Ans'] = str(instance['Answer']).lower()
                
                llama_answer = instance['PEER_LLAMA'][0].split("Answer:")[-1].strip("\n").strip().lower()
                instance["Evaluation"]['PEER_LLAMA_Ans'] = llama_answer if 'Answer:' in instance['PEER_LLAMA'][0] and len(llama_answer) <= 20 else " "
                
                nemo_answer = instance['PEER_NEMO'][0].split("Answer:")[-1].strip("\n").strip().lower()
                instance["Evaluation"]['PEER_NEMO_Ans'] = nemo_answer if 'Answer:' in instance['PEER_NEMO'][0] and len(nemo_answer) <= 20 else " "

                gemma_answer = instance['PEER_GEMMA'][0].split("Answer:")[-1].strip("\n").strip().lower()
                instance["Evaluation"]['PEER_GEMMA_Ans'] = gemma_answer if 'Answer:' in instance['PEER_GEMMA'][0] and len(gemma_answer) <= 20 else " "

            elif task in ["AQuA", "BBH_DU", 'CSQA']:
                instance["Evaluation"]['Correct_Ans'] = str(instance['Answer']).upper()
                
                llama_answer = instance['PEER_LLAMA'][0].split("Answer:")[-1].strip("\n").strip("(").strip(")").strip().upper()
                instance["Evaluation"]['PEER_LLAMA_Ans'] = llama_answer if 'Answer:' in instance['PEER_LLAMA'][0] and len(llama_answer) <= 20 else " "
                
                nemo_answer = instance['PEER_NEMO'][0].split("Answer:")[-1].strip("\n").strip("(").strip(")").strip().upper()
                instance["Evaluation"]['PEER_NEMO_Ans'] = nemo_answer if 'Answer:' in instance['PEER_NEMO'][0] and len(nemo_answer) <= 20 else " "
                
                gemma_answer = instance['PEER_GEMMA'][0].split("Answer:")[-1].strip("\n").strip("(").strip(")").strip().upper()
                instance["Evaluation"]['PEER_GEMMA_Ans'] = gemma_answer if 'Answer:' in instance['PEER_GEMMA'][0] and len(gemma_answer) <= 20 else " "

        except Exception as e:
            with open("errors.txt", "a") as f:
                f.write("Peer Regex: " + str(e) + "\n")
                f.write("Instance: " + str(instance) + "\n")
            retry += 1
            continue
            

        #### Area Chair
        try:
            AC_prompt = AC_prompt.replace('{{Peer_response1}}', instance["Evaluation"]['PEER_LLAMA_Ans']).replace('{{Peer_response2}}', instance["Evaluation"]['PEER_NEMO_Ans']).replace('{{Peer_response3}}', instance["Evaluation"]['PEER_GEMMA_Ans'])
            
            instance['AC_prompt'] = AC_prompt
            _response = gpt_client.chat.completions.create(
                model=args.AC_model,
                messages=[{"role": "system", "content": AC_prompt}],
                temperature=1,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                n=20
            )

            all_responses = [_response.choices[i].message.content for i in
                                    range(len(_response.choices))]
            # all_responses = [x.replace("*", "") for x in all_responses]
            instance['AC_GPT'] = all_responses
            instance['AC_GPT_input_tokens'] = _response.usage.prompt_tokens
            instance['AC_GPT_output_tokens'] = _response.usage.completion_tokens
            
            try:
                if task == 'GSM8k':
                    instance["Evaluation"]['AC_GPT_Ans'] = [x.split("Answer:")[-1].strip("\n").strip().lower() for x in instance['AC_GPT']]
                elif task == "AQuA" or task == "BBH_DU" or task == 'CSQA':
                    instance["Evaluation"]['AC_GPT_Ans'] = [x.split("Answer:")[-1].strip("\n").strip("(").strip(")").strip().upper() for x in instance['AC_GPT']]
            
            except Exception as e:
                with open("errors.txt", "a") as f:
                    f.write("AC Regex: " + str(e) + "\n")

        except Exception as e:
            with open("errors.txt", "a") as f:
                f.write("GPT: " + str(e) + "\n")
            if ("limit" in str(e)):
                time.sleep(3)
            else:
                retry += 1
                continue
        #delete Ratings and scores keys and values from instance
        if task == 'GSM8k':
            try:
                del instance['Ratings']
                del instance['scores']
            except:
                pass

        int_fp = save_fp.replace('.json','_intermediate.jsonl')
        with open(int_fp, 'a') as f:
            json.dump(instance, f, indent=4)
            f.write(',\n')
        time.sleep(2)

        return instance
    return None

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--tasks_list", type=str, nargs="+", default=["AQuA"], help='Supported Tasks: AQuA, BBH_DU, CSQA, GSM8k')
    argparser.add_argument("--task", type=str, default="GSM8k", help='Supported Tasks: AQuA, BBH_DU, CSQA, GSM8k')
    argparser.add_argument("--exp_folder", type=str, default="inference_test",help="Enter the directory name for the experiment results to be stored in")
    argparser.add_argument('--exp_file', type=str,  default='gpt4o_mini_result.json', help="results file name")
    argparser.add_argument('--num_samples', type=int, default=100, help="Number of samples to run")

    argparser.add_argument('--llama_model', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo')#meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'
    argparser.add_argument('--gemma_model', type=str, default='google/gemma-2-9b-it')
    argparser.add_argument('--nemo_model', type=str, default='open-mistral-nemo')
    argparser.add_argument('--AC_model', type=str, default='gpt-4o-mini')
    args = argparser.parse_args()
    
    gpt_client = OpenAI(api_key=OPENAI_API_KEY)
    mist_client = MistralClient(api_key=MISTRAL_API_KEY)
    togetherai_client = Together(api_key=TOGETHER_API_KEY)

    if args.task not in ["AQuA","BBH_DU","CSQA","GSM8k"]:
        print(f"Task {args.task} not supported")
    else:
        for task in tqdm.tqdm(args.tasks_list, desc="Tasks"):
            print(f"Running for {task}")
            args.task = task
            data_dir = os.path.join('Datasets',args.task)
            dataset_fp = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if f.endswith('.json')][0]
            dataset = json.load(open(dataset_fp, encoding="utf-8"))
            random.seed(42)
            random.shuffle(dataset)

            prompt_dir = os.path.join('prompts',args.task)
            prompt_fp = os.path.join(prompt_dir,'peer_prompt.txt')
            ac_prompt_fp = os.path.join(prompt_dir,'AC_prompt.txt')
            prompt = open(prompt_fp).read()
            ac_prompt = open(ac_prompt_fp).read()

            os.makedirs(os.path.join('results',args.task),exist_ok=True)
            res_folder = os.path.join("results",args.task, args.exp_folder)
            os.makedirs(res_folder, exist_ok=True)
            save_fp = os.path.join(res_folder,args.exp_file)
            if os.path.exists(save_fp):
                print(f"Results already exist for {task} in {save_fp}. Skipping...")
                print("Change the exp_file name and run again")
                continue

            new_json = []
            start = time.time()
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(ReFeR, instance, args.task, prompt, ac_prompt, save_fp) for instance in dataset[:args.num_samples]]
                for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                    instance = future.result()
                    if instance:
                        new_json.append(instance)
            end = time.time()
            print(f"Time taken: {end - start} seconds")

            with open(save_fp, 'w') as f:
                json.dump(new_json, f, indent=4)
            
            intermediate_fp = save_fp.replace('.json', '_intermediate.jsonl')
            if os.path.exists(intermediate_fp):
                os.remove(intermediate_fp)

