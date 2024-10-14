import json
import argparse
import tqdm
import time
import os
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *

# suppress warnings
import logging
logging.getLogger().setLevel(logging.ERROR)

def ReFeR(instance):
    source = instance['source']
    system_output = instance['system_output']
    if 'TopicalChat' in args.prompt_peer:
        fact = instance["context"]
        peer_prompt = prompt.replace('{{Document}}', source).replace('{{Response}}', system_output).replace('{{Fact}}', fact)
    else:
        peer_prompt = prompt.replace('{{Document}}', source).replace('{{Summary}}', system_output)
    instance['peer_prompt'] = peer_prompt
    while True:
        # GPT
        try:
            _response = gpt_client.chat.completions.create(
                model=args.GPT_model,
                messages=[{"role": "system", "content": peer_prompt}],
                temperature=1,
                max_tokens=args.max_len,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                n = 20
            )

            all_responses_1 = [_response.choices[i].message.content for i in
                                range(len(_response.choices))]
            instance['Analyze_rate'] = all_responses_1
            instance["Analyze_rate_input_tokens"] = _response.usage.prompt_tokens
            instance["Analyze_rate_output_tokens"] = _response.usage.completion_tokens

        except Exception as e:
            with open("errors.txt", "a") as f:
                f.write("GPT: " + str(e) + "\n")
            if ("limit" in str(e)):
                time.sleep(3)
            else:
                break

        with open(args.save_fp_intermediate, 'a') as f:
            json.dump(instance, f, indent=4)
            f.write(',\n')
        time.sleep(5)
        return instance
    return None

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--prompt_peer', type=str, default='../prompts/TopicalChat/Analyze_Rate/AC_prompt_coh.txt')
    argparser.add_argument('--dataset_fp', type=str, default='../Datasets/TopicalChat/topical_chat.json')
    argparser.add_argument('--save_fp', type=str, default='../results/Topicalchat/inference_test/gpt4o_coh.json')
    argparser.add_argument('--save_fp_intermediate', type=str, default='../results/Topicalchat/inference_test/gpt4o_coh_intermediate.jsonl')
    
    argparser.add_argument('--max_len', type=int, default=64) #For summeval 128 or 96,for topical chat 64
    
    argparser.add_argument('--GPT_model', type=str, default='gpt-4o-mini')#'gpt-3.5-turbo-0613') 
    
    args = argparser.parse_args()
    
    os.makedirs(os.path.dirname(args.save_fp), exist_ok=True)

    gpt_client = OpenAI(api_key=OPENAI_API_KEY)

    dataset = json.load(open(args.dataset_fp))
    prompt = open(args.prompt_peer).read()
    start = time.time()
    new_json = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(ReFeR, instance) for instance in dataset]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            instance = future.result()
            if instance:
                new_json.append(instance)
    end = time.time()
    print(f"Time taken: {end - start} seconds")

    with open(args.save_fp, 'w') as f:
        json.dump(new_json, f, indent=4)

    if os.path.exists(args.save_fp_intermediate):
        os.remove(args.save_fp_intermediate)

