import os
from dotenv import load_dotenv
from groq import Groq
import upload
load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)


def generate_response(prompt):
    data = upload.query_search(prompt)
    print(data)
    input = f"""you are an helpful assistant.
    your task is to answer the user based on the given data. 
    First, you need to understand the data and then answer the user. 
    the following data is extracted from an ebook using vector search. 
    
    Data: {str(data)}
    the data might be unstructured, so you need to understand it and then answer the user.

    User: {prompt}

    Go through the data and answer the user based on the data.
    """
    print(prompt)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": input,
            }
        ],
        model="gemma-7b-it",
    )

    res = chat_completion.choices[0].message.content
    print(res)
    return res