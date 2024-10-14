import json
import argparse
import tqdm
import time
import os
import re
import base64
import requests
from openai import OpenAI
import google.generativeai as genai
from config import *
# suppress warnings
import logging
logging.getLogger().setLevel(logging.ERROR)
os.environ["GRPC_VERBOSITY"] = "NONE"

def extract_rating(text):
    # Regex pattern to match a float number in the format 0 to 5 (including decimals)
    pattern = r"(\b[0-4]\.\d+|\b5(?:\.0+)?|\b[0-5]\b)"
    match = re.search(pattern, text)
    if match:
        return float(match.group())
    return None

def ReFeR(instance):
    # OpenAI API Key
    api_key = OPENAI_API_KEY
    source = instance['Prompt']
    peer_prompt = prompt.replace('{{Input_Prompt}}', source)
    instance['peer_prompt'] = peer_prompt
    image_path = os.path.join(os.path.dirname(args.dataset_fp),f"AGIQA_Images/{instance['Image']}")
    while True:
        # gemini-1.5 flash
        try:
            sample_file = genai.upload_file(path=image_path)
            # Prompt the model with text and the previously uploaded image.
            response = google_model.generate_content([sample_file, peer_prompt])
            instance['PEER_FLASH'] = [response.text]
        except Exception as e:
            with open("errors.txt", "a") as f:
                f.write("FLASH: " + str(e) + "\n")
            if ("limit" in str(e)):
                with open("errors.txt", "a") as f:
                    f.write("FLASH: " + "Sleeping" + "\n")
                time.sleep(3)
            else:
                break

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
                "model": args.GPT_model,
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
            instance['PEER_GPT'] = [response_text]
        except Exception as e:
            with open("errors.txt", "a") as f:
                f.write("PEER GPT: " + str(e) + "\n")
            if ("limit" in str(e)):
                with open("errors.txt", "a") as f:
                    f.write("PEER GPT: " + "Sleeping" + "\n")
                time.sleep(3)
            else:
                break
        
        # Rating Extraction
        for peer in ['PEER_FLASH', 'PEER_GPT']:
            try:
                response = instance[peer][0]
                rating = response.split('Rating:')[-1]
                
                # rating_match = re.search(r'\s*(\d+)', rating)
                rating_match = extract_rating(rating)
                # analysis = response.split("Rating: ")[0]
                if rating_match:
                    instance[f'{peer}_rating'] = str(rating_match)
                else:
                    instance[f'{peer}_rating'] = " "
            except Exception as e:
                print(f"Error: {e}")
                instance[f'{peer}_rating'] = " "
       
        try:
            AC_prompt = ac_prompt.replace('{{Input_Prompt}}', source).replace('{{Peer_response1}}',instance['PEER_FLASH_rating']).replace('{{Peer_response2}}',instance['PEER_GPT_rating'])
            instance['AC_prompt'] = AC_prompt
            payload = {
                "model": args.AC_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": AC_prompt,
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
                "max_tokens": args.max_len + 128,
                "temperature": 1,  
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "n": 20
            }
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            response_data = response.json()
            instance['AC_GPT'] = [response_data['choices'][i]['message']['content'] for i in range(len(response_data['choices']))]
        except Exception as e:
            with open("errors.txt", "a") as f:
                f.write("AC GPT: " + str(e) + "\n")
            if ("limit" in str(e)):
                with open("errors.txt", "a") as f:
                    f.write("AC GPT: " + "Sleeping" + "\n")
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
    argparser.add_argument('--prompt_peer', type=str,  default='prompts/AGIQA/peer_prompt.txt')
    argparser.add_argument('--prompt_ac', type=str,  default='prompts/AGIQA/AC_prompt.txt')
    argparser.add_argument('--dataset_fp', type=str, default='Datasets/AGIQA/AGIQA_Dataset_500.json')
    argparser.add_argument('--save_fp', type=str,  default='results/AGIQA/ReFeR/gpt4o_select_500.json')
    argparser.add_argument('--save_fp_intermediate', type=str,  default='results/AGIQA/ReFeR/gpt4o_select_500.jsonl')
    
    argparser.add_argument('--max_len', type=int, default=128)
    
    argparser.add_argument('--GPT_model', type=str, default='gpt-4o-mini')
    argparser.add_argument('--gemini_model', type=str, default='gemini-1.5-flash')
    argparser.add_argument('--AC_model', type=str, default='gpt-4o-2024-08-06')
    
    args = argparser.parse_args()
    
    os.makedirs(os.path.dirname(args.save_fp), exist_ok=True)

    gpt_client = OpenAI(api_key=OPENAI_API_KEY)
    genai.configure(api_key=GOOGLE_API_KEY)
    google_model = genai.GenerativeModel(args.gemini_model)

    dataset = json.load(open(args.dataset_fp))
    prompt = open(args.prompt_peer).read()
    ac_prompt = open(args.prompt_ac).read()

    if os.path.exists(args.save_fp):
        print(f"File {args.save_fp} already exists. Exiting.")
        exit()

    new_json = []
    for instance in tqdm.tqdm(dataset[:500]):
        result = ReFeR(instance)
        if result:
            new_json.append(result)


    with open(args.save_fp, 'w') as f:
        json.dump(new_json, f, indent=4)

    if os.path.exists(args.save_fp_intermediate):
        os.remove(args.save_fp_intermediate)

