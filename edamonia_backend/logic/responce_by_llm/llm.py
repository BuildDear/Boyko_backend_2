from litellm import completion

def generate_response(question, context):
    prompt = (
        f"Ви є асистентом, який відповідає виключно на основі наданого контексту.\n\n"
        f"Правила:\n"
        f"- Використовуйте лише інформацію, надану в контексті, для формування відповіді. Ігноруйте будь-які попередні знання чи припущення.\n"
        f"- Якщо відповідь явно присутня в контексті, надайте чітку та точну відповідь.\n"
        f"- Якщо відповідь не може бути знайдена в контексті, відповідайте лише: 'Вибачте, я не зміг знайти відповідь у наданому контексті.'\n"
        f"- Не здогадуйтесь, не припускайте та не вигадуйте інформацію. Дотримуйтесь строго меж наданого контексту.\n"
        f"- Контекст: {context}\n"
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
        generated_text += str(chunk["choices"][0]['delta']['content'])

    return generated_text
