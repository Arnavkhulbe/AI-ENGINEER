import os
from dotenv import load_dotenv
from groq import Groq

# Load API key from .env
load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key not found!")

# Create Groq client
client = Groq(api_key=my_api_key)

# ---------------- Settings ----------------

model = "llama-3.3-70b-versatile"
temperature = 1


# ------------------------------------------

system_message = {
    "role": "system",
    "content": "You are an expert career name giver..give only five name "
}

user_message = {
    "role": "user",
    "content": "i have a app where i scan people body for fitness through ai..give me comapany name"
}

response = client.chat.completions.create(
    model=model,
    temperature=temperature, #setting randomness
    messages=[system_message, user_message]
)

print("======================================")
print(response.choices[0].message.content)
print("======================================")