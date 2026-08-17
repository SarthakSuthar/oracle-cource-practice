# ReAct (Reason -> Act-> observ)

import math

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama.chat_models import ChatOllama

model = ChatOllama(model="qwen2.5:0.5b", validate_model_on_init=True)


@tool
def add(a: float, b: float) -> float:
    """
    Add two numbers.
    Agent will use this when it detects an addition.
    """
    return a + b


@tool
def subtract(a: float, b: float) -> str:
    """
    Subtract the second number from the first.
    Used for subtraction task.
    """
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers.
    Used for multiplication task.
    """
    return a * b


@tool
def divide(a: float, b: float) -> str:
    """
    Divide the first number by the second.
    Includes error handling for division by zero.
    """
    if b == 0:
        return "Error : Can not divide by zero."

    return str(a / b)


@tool
def sqrt(num: float) -> float:
    """
    Calculate the square root of a number.
    Includes error handling for negative inputs.
    """
    if num < 0:
        return "Error : Cannot take square root of a negative number"
    return str(math.sqrt(num))


tools = [add, subtract, multiply, divide, sqrt]

print("=== All tools ===")
for t in tools:
    print(f" ->  {t.name} : {t.description}")
print()

system_prompt = """
You are a mathematical reasoning agent.

You have access to tools for addition, subtraction, multiplication,
division, and square root.

IMPORTANT RULES:

1. Perform calculations in the exact order specified by the user.
2. When a calculation contains multiple sequential operations,
   use the result of the previous operation as the input to the next tool.
3. Never execute dependent calculations independently.
4. For example:

   User: 15 multiplied by 8, then divided by 3

   Correct:
   multiply(15, 8) -> 120
   divide(120, 3) -> 40

5. For:
   42 + 5 - 10

   Correct:
   add(42, 5) -> 47
   subtract(47, 10) -> 37

6. Do not perform arithmetic yourself when an appropriate tool exists.
7. Do not call multiple dependent tools using the original user inputs.
8. After receiving a tool result, use that result when calling the next
   dependent tool.
"""


agent = create_agent(model=model, tools=tools, system_prompt=system_prompt)


def run_agent(question: str):
    """ "Run the agent and print clean beginner friendly execution here"""

    print(f"\n User : {question}")
    print("-" * 60)

    result = agent.invoke({"messages": [("user", question)]})

    print("Clean gent Execution trace.")
    print("-" * 60)

    step = 1

    for msg in result["messages"]:
        if msg.type == "human":
            print(f"{step}. User asked:")
            print(f"    {msg.content}")
            step += 1

        elif msg.type == "ai" and getattr(msg, "tool_calls", None):
            for tool_call in msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                print(f"{step}. Agent decision:")
                print(f"I need to use the tool: {tool_name}")
                print(f"Tool input: {tool_args}")
                step += 1

        elif msg.type == "tool":
            print(f"{step}. Tool observation:")
            print(f"Tool returned: {msg.content}")
            step += 1

        elif msg.type == "ai" and msg.content:
            print(f"{step}. Final answer: ")
            print(f"{msg.content}")
        step += 1

    print("=" * 60)


run_agent("What is 42 + 5 - 10?")

run_agent("What is 15 multiplied by 8, then divided by 3?")

run_agent(
    "I have a rectangle with width 12 and height 7. "
    "What is its area, and what is the square root of that area?"
)

run_agent("What is 100 divided by 0?")

print("Agent demo complete!")
