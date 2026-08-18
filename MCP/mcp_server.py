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


@mcp.tool
def divide(a: float, b: float) -> str:
    """Divide the first number by the second.
    Includes error handling for division by zero.
    Use for division operation.
    """
    if b == 0:
        return "Zero can not be divided"
    return str(a / b)


if __name__ == "__main__":
    mcp.run(transport="stdio")
