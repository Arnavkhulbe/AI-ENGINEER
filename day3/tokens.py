import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key kaha hai bhai")

client = Groq(api_key=my_api_key)
role = "user"

# 3 prompts
prompt1 = "Hi!"
prompt2 = "Explain time travel in Detail"
prompt3 = "Write a 1000 word essay on Machine Learning"

prompts = [prompt1, prompt2, prompt3]

for prompt in prompts:

    message = {
        "role": role,
        "content": prompt
    }

    messages = [message]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )

    usage = response.usage  #Get the token usage information from the response and store it in the variable usage.

    print(
        f"Prompt: {prompt} --> "
        f"prompt_tokens: {usage.prompt_tokens} "
        f"completion_tokens: {usage.completion_tokens} "
        f"total_tokens: {usage.total_tokens}"
    )

    print("-" * 60)
    print(response.choices[0].message.content)
    print("-" * 60)
    print()