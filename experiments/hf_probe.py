import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"

def make_prompt(a ,b):
    return f"{a} + {b} ="


def extract_number(text):
    # TODO: use regex to extract the number from the text
    # return int() or None 
    match = re.search(r'\d+', text)
    if match:
        return int(match.group())
    return None

def reward_function(response_text, target):
    response_number = extract_number(response_text)
    print(f"Target: {target}")
    if response_number is None:
        return 0

    return 1 if response_number == target else 0

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)


    model.eval()


    prompts = [
        (7, 5),
        (10, 15),
        (20, 30),
        (100, 200),
        ]

    
    for a, b in prompts: 
        prompt = make_prompt(a, b)
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][prompt_len:]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        target = a + b

        reward = reward_function(response_text, target)
        print(f"Prompt: {prompt}")
        print(f"Model response: {response_text}")
        print(f"Reward: {reward}")
        print("-" * 30)


if __name__ == "__main__":
    main()
     
