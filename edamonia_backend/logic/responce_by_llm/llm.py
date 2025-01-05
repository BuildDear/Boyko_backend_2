from litellm import completion

def generate_response(question, context):
    prompt = (
        f"Ти є ввічливим українським асистентом, який ввічливо відповідає на основі переданих знань.Чітко, розгорнуто та ввічливо відповідай на питання.\n\n"
        f"Правила:\n"
        f"- Не згадуй ні за які питання, лише розгорнуто та чітко давай відповід.\n"
        f"- Використовуйте лише інформацію, наданих занань, для формування відповіді.\n"
        f"- Якщо відповідь явно присутня в знаннях, надайте чітку та точну відповідь.\n"
        f"- Якщо відповідь не може бути знайдена в контексті, відповідайте лише: 'Вибачте, я не зміг знайти відповідь у наданому контексті.'\n"
        f"- Не здогадуйтесь, не припускайте та не вигадуйте інформацію. Дотримуйтесь строго меж наданого контексту.\n"
        f"- Додаткові знання: {context}\n"
    )

    response = completion(
        model="groq/llama3-8b-8192",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
        stream=True,
    )

    generated_text = ""
    for chunk in response:
        content = chunk["choices"][0]['delta'].get('content', None)
        if content:
            generated_text += content

    return generated_text.strip()
