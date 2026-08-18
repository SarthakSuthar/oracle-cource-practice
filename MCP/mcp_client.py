import asyncio
import os
import sys

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama.chat_models import ChatOllama


async def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_dir = os.path.join(current_dir, "mcp_server.py")

    client = MultiServerMCPClient(
        {
            "math": {
                "command": sys.executable,
                "args": [server_dir],
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()
    print("=" * 55)
    print("MCP Agent - Tools discovered from MCP Server")
    print("=" * 55)
    print(f"\n Found {len(tools)} tools from MCP server:\n")
    for t in tools:
        print(f" {t.name}: {t.description[:60]}...")
    print()

    model = ChatOllama(model="qwen2.5:0.5b", validate_model_on_init=True)

    sysytem_prompt = """
    You are a math problem solver, always respond with mathemaical breakdown and answer only. Never ask further Questions.
"""

    agent = create_agent(model=model, tools=tools, system_prompt=sysytem_prompt)

    async def run_agent(question: str):
        print(f"\n User : {question}")
        print("=" * 60)

        result = await agent.ainvoke({"messages": [("user", question)]})

        for msg in result["messages"]:
            if msg.type == "human":
                continue
            elif msg.type == "ai":
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"Agent thinks -> calling: {tc['name']}({tc['args']})")
                elif msg.content:
                    print(f"Agent Answer: {msg.content}")
            elif msg.type == "tool":
                content = msg.content
                if isinstance(content, list):
                    texts = [
                        item["text"]
                        for item in content
                        if isinstance(item, dict) and "text" in item
                    ]
                    content = ", ".join(texts) if texts else str(content)
                print(f"Tool result: {content}")

        print("=" * 60)
        print()

    await run_agent("What is 42 + 5?")

    await run_agent("What is 15 multiplied by 8, then result divided by 3?")

    await run_agent("What is 100 divide by 0?")


if __name__ == "__main__":
    asyncio.run(main())
