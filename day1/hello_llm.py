import os #Used here to access environment variables.
from dotenv import load_dotenv #Import the function that loads environment variables from a .env file.
from groq import Groq  #From the groq library, import the Groq class...Import the class used to create a Groq client and interact with Groq's AI models.
load_dotenv() #It reads the .env file and loads its variables into your program's environment.

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key kaha hai bhai")

client = Groq(api_key=my_api_key)#creates client and  Now your client is connected to Groq's API, and you can use it to send requests.


#It sends your prompt to the AI model and gets the AI's response.
#client- Uses the Groq client you created.
#chat- "I want to use the Chat API."
#complete-"Generate (complete) the assistant's next reply."
#create- Actually sends the request to Groq.
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": "Do you know Padho with Pratyush?"
        }
    ]
)

print(response)