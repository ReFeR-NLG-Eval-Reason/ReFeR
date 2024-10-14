import json
import argparse
import tqdm
import time
import os
import re
from openai import OpenAI
import google.generativeai as genai
from config import *
# suppress warnings
import logging
logging.getLogger().setLevel(logging.ERROR)
os.environ["GRPC_VERBOSITY"] = "NONE"


def ReFeR(instance):
    source = instance['CAPTION']
    peer_prompt = prompt.replace('{{Caption}}', source)
    instance['peer_prompt'] = peer_prompt
    file_path = os.path.join(args.image_dir,f"{instance['IMAGE_KEY']}.jpg")
    while True:
        # gemini-1.5 flash
        try:
            sample_file = genai.upload_file(path=file_path)
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
        try:
            response = gpt_client.chat.completions.create(
                model=args.GPT_model,
                messages=[
                    {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": peer_prompt},
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": instance['OriginalURL'],
                        },
                        },
                    ],
                    }
                ],
                temperature=1,
                max_tokens=args.max_len+64,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                
            )
            instance['PEER_GPT'] = [response.choices[0].message.content]

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
                rating_match = re.search(r'\s*(\d+)', rating)
                analysis = response.split("Rating: ")[0]
                if rating_match:
                    instance[f'{peer}_rating'] = rating_match.group(1).rstrip()
                else:
                    instance[f'{peer}_rating'] = " "
            except:
                instance[f'{peer}_rating'] = response

        
        #### Area Chair
        try:
            AC_prompt = ac_prompt.replace('{{Caption}}', source).replace('{{Peer_LLM1_Rating}}',instance['PEER_FLASH_rating']).replace('{{Peer_LLM2_Rating}}',instance['PEER_GPT_rating'])
            instance['AC_prompt'] = AC_prompt
            _response = gpt_client.chat.completions.create(
                model=args.AC_model,
                messages=[
                    {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": peer_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": instance['OriginalURL'],
                            },
                        },
                    ],
                    }
                ],
                temperature=1,
                max_tokens=args.max_len + 128,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                n=20
            )

            all_responses = [_response.choices[i].message.content for i in
                                    range(len(_response.choices))]
            instance['AC_GPT'] = all_responses
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
    argparser.add_argument('--prompt_peer', type=str,  default='prompts/ICQD/peer_prompt.txt') #path to the peer prompt
    argparser.add_argument('--prompt_ac', type=str,  default='prompts/ICQD/AC_prompt.txt') #path to the AC prompt
    argparser.add_argument('--dataset_fp', type=str, default='Datasets/ICQD/Image_Caption_Quality_Dataset.json') #path to the dataset
    argparser.add_argument('--image_dir', type=str, default='Datasets/ICQD/ICQD_Images') #directory containing the images
    argparser.add_argument('--save_fp', type=str,  default='results/ICQD/ReFeR/gpt4o_rating.json') #path to save the final results
    argparser.add_argument('--save_fp_intermediate', type=str,  default='results/ICQD/ReFeR/gpt4o_rating_intermediate.jsonl') #intermediate file is used to save the results as we go along to avoid losing all the results if the program crashes
    
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
    for instance in tqdm.tqdm(dataset):
        result = ReFeR(instance)
        if result:
            new_json.append(result)

    with open(args.save_fp, 'w') as f:
        json.dump(new_json, f, indent=4)
    
    # Remove the intermediate file
    if os.path.exists(args.save_fp_intermediate):
        os.remove(args.save_fp_intermediate)
