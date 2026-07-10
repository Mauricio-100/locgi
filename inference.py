import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

completion = client.chat.completions.create(
    model="zai-org/GLM-5.2:featherless-ai",
    messages=[
        {
            "role": "user",
            "content": "tu est gopu.inc cree par Mauricio-100 voila le question de l'user: tu est qui ??"
        }
    ],
)

print(completion.choices[0].message)
