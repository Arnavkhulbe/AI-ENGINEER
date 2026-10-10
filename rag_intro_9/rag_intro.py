import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")


if not my_api_key:
    raise ValueError("api key kaha hai bhai ")
client=Groq(api_key=my_api_key)
model="llama-3.3-70b-versatile"
# step 1
knowledge_base = {
    "age": "The age of pratyush is 25 years",
    "net_worth": "The net worth of pratyush is 2000"
}


# step 2 retrieval
def retrieve_info(question):
    question = question.lower()

    if "age" in question:
        return knowledge_base["age"]

    elif "net worth" in question:
        return knowledge_base["net_worth"]

    else:
        return None
def ask_llm(question):
    context=retrieve_info(question )
    message={
        "role":"user",
        "content":question
    } 
    sys_prompt=f"""answer in one line only .answer onyl based on this content.dont hallucinate context={context}"""
    system_message={
        "role":"system",
        "content":sys_prompt
    }
    
    messages=[system_message,message]
    response=client.chat.completions.create(
        model=model,
        messages=messages)
    answer = response.choices[0].message.content
    return answer
question="do you know bhaskar khulbe"
print(ask_llm(question))
    
