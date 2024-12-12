import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "BSC-LT/salamandra-2b-instruct"


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
    """
    Генерує відповідь на будь-яке запитання, використовуючи модель.
    """
    prompt = (
        f"You are a highly intelligent and helpful assistant. Answer the following question in a clear and concise manner:\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Генерація відповіді
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=64,  # Обмеження довжини відповіді
        temperature=0.4,  # Контроль креативності
        do_sample=True,  # Використання семплювання для варіативності
    )

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response.strip()

