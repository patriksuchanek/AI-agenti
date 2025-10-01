import os
import json
from datetime import datetime
import pytz
from pprint import pprint
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(os.environ.get("OPENAI_API_KEY"),


tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate_sum",
            "description": "Sečte dvě čísla.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "První číslo"},
                    "b": {"type": "number", "description": "Druhé číslo"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Vrátí aktuální čas pro zadané časové pásmo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Název časového pásma, např. 'Europe/Prague' nebo 'Asia/Manila'.",
                    }
                },
                "required": ["timezone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "string_reverse",
            "description": "Vrátí obrácený text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text, který se má obrátit pozpátku",
                    }
                },
                "required": ["text"],
            },
        },
    },
]


def calculate_sum(a, b):
    return a + b

def get_current_time(timezone: str):
    try:
        tz = pytz.timezone(timezone)
        return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        return f"Neplatné časové pásmo: {e}"

def string_reverse(text: str):
    return text[::-1]


response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Jsi chytrý AI asistent."},
        {"role": "user", "content": "Můžeš mi prosím obrátit text 'Ahoj světe' pozpátku?"},
    ],
    tools=tools,
    tool_choice="auto",
)

print("--- First response ---")
print(response.to_json())

message = response.choices[0].message


if message.tool_calls:
    tool_call = message.tool_calls[0]
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    if function_name == "calculate_sum":
        result = calculate_sum(**arguments)
    elif function_name == "get_current_time":
        result = get_current_time(**arguments)
    elif function_name == "string_reverse":
        result = string_reverse(**arguments)
    else:
        result = "Unknown function"


    followup = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Jsi chytrý AI asistent."},
            {"role": "user", "content": "Můžeš mi prosím obrátit text 'Ahoj světe' pozpátku?"},
            message,  
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            },
        ],
    )

    print("--- Final response ---")
    print(followup.choices[0].message.content)











