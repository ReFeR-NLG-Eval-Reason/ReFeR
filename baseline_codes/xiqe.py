import json
import argparse
import tqdm
import time
import os
import re
import base64
import requests
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OPENAI_API_KEY

from openai import OpenAI


# suppress warnings
import logging
logging.getLogger().setLevel(logging.ERROR)
os.environ["GRPC_VERBOSITY"] = "NONE"
API_KEY = OPENAI_API_KEY

def extract_rating(text):
    # Regex pattern to match a float number in the format 0 to 5 (including decimals)
    pattern = r"(\b[0-4]\.\d+|\b5(?:\.0+)?|\b[0-5]\b)"
    match = re.search(pattern, text)
    if match:
        return float(match.group())
    return None

def ReFeR(instance):
    # OpenAI API Key
    api_key = API_KEY
    source = instance['Prompt']
    peer_prompt = prompt.replace('{{Input_Prompt}}', source)
    instance['peer_prompt'] = peer_prompt
    image_path = os.path.join(os.path.dirname(args.dataset_fp),f"AGIQA_Images/{instance['Image']}")
    while True:

        # Function to encode the image
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        # Getting the base64 string
        base64_image = encode_image(image_path)

        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
        }
        try:
            payload = {
                "model": args.AC_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": peer_prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": args.max_len,
                "temperature": 1,  
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
            }

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            response_data = response.json()
            response_text = response_data['choices'][0]['message']['content']
            try:
                response_text = response_text.strip('\n').strip('```').strip('json').strip()
                response_text = json.loads(response_text)
                Analysis = response_text['Alignment analysis']
                rating = response_text['Alignment score']
            except:
                Analysis = response_text
                rating = ''
                print(response_text)
            Final_response = 'Analysis: ' + Analysis + ' Rating: ' + rating.split('/')[0]
            instance['AC_GPT'] = [Final_response]
        except Exception as e:
            with open("errors.txt", "a") as f:
                f.write("PEER GPT: " + str(e) + "\n")
            if ("limit" in str(e)):
                with open("errors.txt", "a") as f:
                    f.write("PEER GPT: " + "Sleeping" + "\n")
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
    argparser.add_argument('--prompt_peer', type=str,  default='../prompts/AGIQA/xiqe_prompt.txt')
    argparser.add_argument('--dataset_fp', type=str, default='../Datasets/AGIQA/AGIQA_Dataset_500.json')
    argparser.add_argument('--save_fp', type=str,  default='../results/AGIQA/XIQE/gpt4o_select_500.json')
    argparser.add_argument('--save_fp_intermediate', type=str,  default='../results/AGIQA/XIQE/gpt4o_select_500.jsonl')
    
    argparser.add_argument('--max_len', type=int, default=128)
    argparser.add_argument('--AC_model', type=str, default='gpt-4o-2024-08-06')
    
    args = argparser.parse_args()
    
    os.makedirs(os.path.dirname(args.save_fp), exist_ok=True)

    gpt_client = OpenAI(api_key=OPENAI_API_KEY)
    dataset = json.load(open(args.dataset_fp))
    prompt = open(args.prompt_peer).read()

    new_json = []
    for instance in tqdm.tqdm(dataset[:500]):
        result = ReFeR(instance)
        if result:
            new_json.append(result)


    with open(args.save_fp, 'w') as f:
        json.dump(new_json, f, indent=4)
    
    if os.path.exists(args.save_fp_intermediate):
        os.remove(args.save_fp_intermediate)
