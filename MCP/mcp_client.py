import os

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from LangChain.core_building_blocks import ChatOllama


async def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_dir = os.path.join(current_dir, "mcp_server.py")

    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
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

    agent = create_agent(model=model, tools=tools)

    async def run_agent(question : str):
        
