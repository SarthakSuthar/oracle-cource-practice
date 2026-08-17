from fastmcp import FastMCP

mcp = FastMCP(name="Math")


@mcp.tool
def add(a: float, b: float) -> float:
    """Add two numbers together, Use for addition operator."""
    return a + b


@mcp.tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together, Use for multiplication operator."""
    return a * b


if __name__ == "__main__":
    mcp.run(transport="stdio")
