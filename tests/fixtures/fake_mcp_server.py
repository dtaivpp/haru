"""A tiny stdio MCP server for tests: one read-only tool, one write tool, one failing tool."""
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

server = MCPServer("fake")


@server.tool(description="Look up a repo's star count", annotations=ToolAnnotations(read_only_hint=True))
def lookup(repo: str) -> str:
    return f"{repo} has 42 stars"


@server.tool(description="Delete a repository")
def delete_repo(repo: str) -> str:
    return "deleted"


@server.tool(description="Always fails", annotations=ToolAnnotations(read_only_hint=True))
def broken() -> str:
    raise ToolError("upstream exploded")  # surfaced to the client as is_error with this text


if __name__ == "__main__":
    server.run()
