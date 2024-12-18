import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv

load_dotenv()
print(os.getenv("HF_TOKEN"))

model_id = "meta-llama/Llama-3.2-1B"


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)

tokenizer.chat_template = "default"

def generate_assistant_response(question: str):

    print(question, "question")
    inputs = tokenizer.encode(question, return_tensors="pt").to(model.device)
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.3,
        top_p=0.7
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt from the response
    response = response.replace(question, "").strip()

    return response

