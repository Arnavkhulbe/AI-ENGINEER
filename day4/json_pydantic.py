import os
import json
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel # here basemodel acts as a parent class for the ticket , so ticket automaticaly gets usefull features like tye conversion, data validation

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key kaha hai bhai")

client = Groq(api_key=my_api_key)

model = "llama-3.3-70b-versatile"


class Ticket(BaseModel): #  Defines the structure of a ticket.
    name: str
    email: str
    issue: str

# Generate a JSON Schema from the Ticket class so the AI knows
# what fields (name, email, issue) are expected in the output.
schema = Ticket.model_json_schema()

response_format = {   #Tell the AI to return its response as a JSON object instead of plain text.
    "type": "json_object"
}
# Create the system prompt that instructs the AI what to do.
# {schema} inserts the JSON Schema into this prompt using an f-string.
system_prompt = f"""
Extract the personal information from the ticket strictly based on the schema below and return only valid JSON.

Schema:
{schema}
"""

message_system = {
    "role": "system",
    "content": system_prompt
}

text = """
Hello, my name is Arnav.
My iPhone is not working.
My address is Noida.
My email is abc@gmail.com.
My contact number is 1234.
"""

prompt = f"""
This is a customer support ticket.

{text}
"""

message = {
    "role": "user",
    "content": prompt
}

messages = [message_system, message]

response = client.chat.completions.create(
    model=model,
    messages=messages,
    response_format=response_format
)

answer = response.choices[0].message.content

print(answer)


# Convert the AI's JSON response into a Python dictionary.
data = json.loads(answer)

# Create a Ticket object by unpacking the dictionary into keyword arguments.
ticket = Ticket(**data)

print(ticket.name)
print(ticket.email)
print(ticket.issue)