from dotenv import load_dotenv
from groq import Groq
import os
import re

# Load variables from .env file
load_dotenv()

# Get API key
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key missing!")

# Create Groq client
client = Groq(api_key=my_api_key)

# LLM model to use
model = "llama-3.3-70b-versatile"


# ---------------- TOOLS ---------------- #

# Tool 1 : Returns price of a product
def get_product_price(product):
    prices = {
        "iPhone 17": 80000,
        "Samsung S25": 70000
    }
    return prices.get(product, "Product not found")


# Tool 2 : Simple calculator
def calculator(expression):
    try:
        return eval(expression)
    except:
        return "Invalid expression"


# Dictionary that maps tool names to actual Python functions.
# The AI returns only the tool name as text.
# We use this dictionary to find and execute the correct function.
tools = {
    "get_product_price": get_product_price,
    "calculator": calculator
}


# Instructions given to the AI
system_prompt = """
You are a shopping assistant.

Available tools:

1. get_product_price(product)
2. calculator(expression)

When you need a tool, reply exactly like:

Thought: ...
Action: tool_name(argument)

When you know the answer:

Final Answer: ...
"""


# ---------------- AGENT ---------------- #

def run_agent(question):

    # Conversation sent to the LLM
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": question
        }
    ]

    # Maximum 5 reasoning steps
    for step in range(5):

        print(f"\n------ STEP {step+1} ------")

        # Send conversation to the LLM
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0
        )

        # Extract only the text response
        answer = response.choices[0].message.content

        print(answer)

        # Stop if the AI has produced the final answer
        if "Final Answer:" in answer:
            final = answer.split("Final Answer:")[-1].strip()
            return final

        # Extract tool name and tool input from the AI response
        match = re.search(
            r"Action:\s*(\w+)\((.*?)\)",
            answer
        )

        # AI forgot to call a tool
        if not match:
            print("No action found.")
            break

        # Extract values from regex
        tool_name = match.group(1)
        tool_input = match.group(2).strip("\"'")

        print("Tool :", tool_name)
        print("Input:", tool_input)

        # Find the correct Python function
        tool = tools[tool_name]

        # Execute the function
        observation = tool(tool_input)

        print("Observation:", observation)

        # Save AI response in conversation memory
        messages.append({
            "role": "assistant",
            "content": answer
        })

        # Give tool result back to the AI
        messages.append({
            "role": "user",
            "content": f"Observation: {observation}"
        })


# Start the agent
question = "I have 5000 rupees. Can I buy an iPhone 17?"
result = run_agent(question)

print("\nFinal Result:", result)