import os
from dotenv import load_dotenv
import litellm

load_dotenv("code_helper/.env")

print("NVIDIA_API_KEY set:", bool(os.getenv("NVIDIA_API_KEY")))
print("NVIDIA_BASE_URL:", os.getenv("NVIDIA_BASE_URL"))

try:
    resp = litellm.completion(
        model="nvidia_nim/meta/llama-3.2-11b-vision-instruct",
        messages=[{"role": "user", "content": "hi"}],
        timeout=10,
    )
    print("Success:", resp.choices[0].message.content)
except Exception as e:
    print("Error:", type(e), e)
