import json
import argparse
import tqdm
import time
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import google.generativeai as genai #pip install -q -U google-generativeai
from together import Together
from config import *
# suppress warnings
import logging
logging.getLogger().setLevel(logging.ERROR)

togetherai_client = Together(api_key=TOGETHER_API_KEY)

def ReFeR(instance):
    source = instance['source']
    system_output = instance['system_output']
    fact = instance['context']
    peer_prompt = prompt.replace('{{Document}}', source).replace('{{Response}}', system_output).replace('{{Fact}}', fact)
    instance['peer_prompt'] = peer_prompt
    AC_prompt = prompt.replace('{{Document}}', source).replace('{{Response}}', system_output).replace('{{Fact}}', fact)
    while True:
        # llama
        try:
            _response = togetherai_client.chat.completions.create(messages=[
                            {
                                "role": "user",
                                "content": peer_prompt,
                            }], model=args.llama_model,
                            # max_tokens=5
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
        
        # Mistral-8x7b
        try:
            messages = [ChatMessage(role="user", content=peer_prompt)]
            _response = mist_client.chat(
                model=args.nemo_model,
                messages=messages,
                temperature=1.0,
                max_tokens=args.max_len,
                # max_tokens=5,
                top_p=1.0
            )
            all_responses_2 = [_response.choices[i].message.content for i in
                            range(len(_response.choices))]
            instance['PEER_NEMO'] = all_responses_2
            instance['PEER_NEMO_input_tokens'] = _response.usage.prompt_tokens
            instance['PEER_NEMO_output_tokens'] = _response.usage.completion_tokens
        except Exception as e:
            with open("errors.txt", "a") as f:
                f.write("Nemo: " + str(e) + "\n")
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
                            # max_tokens=5
                            )

            all_responses_3 = [_response.choices[i].message.content for i in
                                range(len(_response.choices))]
            all_responses_3 = [x.replace("*", "") for x in all_responses_3]
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

        # Rating Extraction
        for peer in ['PEER_LLAMA', 'PEER_NEMO', 'PEER_GEMMA']:
            try:
                response = instance[peer][0]
                response = response.split('Rating:')[-1]
                rating_match = re.search(r'\s*([\d\.]+)', response)
                
                if rating_match:
                    
                    rating_str = float(rating_match.group(1).rstrip('.'))
                    category = ''
                    if 'gro' in args.prompt_peer:
                        if rating_str >= 0.7: #2
                            category = 'high'
                        elif 0.35 < rating_str < 0.7:#1.5, 2
                            category = 'medium'
                        else:
                            category = 'low'
                    else:
                        if rating_str >= 2: #0.7
                            category = 'high'
                        elif 1.5 < rating_str < 2:#0.35, 0.7
                            category = 'medium'
                        else:
                            category = 'low'
                    instance[f'{peer}_summary'] = category + "(" + str(round(rating_str,2)) + ")"
                else:
                    instance[f'{peer}_summary'] = " "
            except:
                instance[f'{peer}_summary'] = " "

        
        #### Area Chair
        try:
            AC_prompt = ac_prompt.replace('{{Document}}', source).replace('{{Response}}', system_output).replace('{{Fact}}', fact).replace('{{Peer_response1}}', instance['PEER_LLAMA_summary']).replace('{{Peer_response2}}', instance['PEER_NEMO_summary']).replace('{{Peer_response3}}', instance['PEER_GEMMA_summary'])
            instance['AC_prompt'] = AC_prompt
            _response = gpt_client.chat.completions.create(
                model=args.AC_model,
                messages=[{"role": "system", "content": AC_prompt}],
                temperature=1,
                max_tokens=args.max_len + 128,
                # max_tokens = 5,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                n=1
            )

            all_responses = [_response.choices[i].message.content for i in
                                    range(len(_response.choices))]
            instance['AC_GPT'] = all_responses
            instance['AC_GPT_input_tokens'] = _response.usage.prompt_tokens
            instance['AC_GPT_output_tokens'] = _response.usage.completion_tokens

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
        time.sleep(2)
        return instance
    return None

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--prompt_peer', type=str,  default='prompts/TopicalChat/Peer_Prompts/peer_prompt_coh.txt')
    argparser.add_argument('--prompt_ac', type=str,  default='prompts/TopicalChat/AC_Prompts/AC_prompt_coh.txt')
    argparser.add_argument('--topical_fp', type=str, default='Datasets/TopicalChat/topical_chat.json')
    argparser.add_argument('--save_fp', type=str,  default='results/TopicalChat/ReFeR_test/gpt4o_mini_coh.json')
    argparser.add_argument('--save_fp_intermediate', type=str,  default='results/TopicalChat/ReFeR_test/gpt4o_mini_coh_intermediate.jsonl')
    
    argparser.add_argument('--max_len', type=int, default=128)
    
    argparser.add_argument('--llama_model', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo')#meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'
    argparser.add_argument('--gemma_model', type=str, default='google/gemma-2-9b-it')
    argparser.add_argument('--nemo_model', type=str, default='open-mistral-nemo')
    argparser.add_argument('--AC_model', type=str, default='gpt-4o-mini')#'gpt-3.5-turbo-0613')
    

    args = argparser.parse_args()
    
    os.makedirs(os.path.dirname(args.save_fp), exist_ok=True)

    if os.path.exists(args.save_fp):
        print(f"File {args.save_fp} already exists. Exiting.")
        exit()

    gpt_client = OpenAI(api_key=OPENAI_API_KEY)
    mist_client = MistralClient(api_key=MISTRAL_API_KEY)
   
    topical = json.load(open(args.topical_fp))
    prompt = open(args.prompt_peer).read()
    ac_prompt = open(args.prompt_ac).read()

    new_json = []
    start = time.time()
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(ReFeR, instance) for instance in topical]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            instance = future.result()
            if instance:
                new_json.append(instance)

    with open(args.save_fp, 'w') as f:
        json.dump(new_json, f, indent=4)
    
    if os.path.exists(args.save_fp_intermediate):
        os.remove(args.save_fp_intermediate)

    end = time.time()
    print(f"Time taken: {end - start} seconds")
    # time.sleep(15)