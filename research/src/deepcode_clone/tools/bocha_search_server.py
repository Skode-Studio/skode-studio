

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP



load_dotenv()


# Initialize FastMCP Server
mcp = FastMCP(
  "bocha-search-mcp",
  prompt="""
  # Bocha Search MCP Server
  
  Bocha is a Chinese search engine for AI, This server provides tools for searching the web using Bocha Search API.
  It allows you to get enhanced search details from billions of web documents, including weather, news, wikis, healthcare, train tickets, images, and more.

  ## Available Tools

  ### 1. bocha_web_search
  Search with Bocha Web Search and get enhanced search details from billions of web documents, including page titles, urls, summaries, site names, site icons, publication dates, image links, and more.

  ### 2. bocha_ai_search
  Search with Bocha AI Search, recognizes the semantics of search terms and additionally returns structured modal cards with content from vertical domains.

  ## Output Format

  All search results will be formatted as text with clear sections for each
  result item, including:

  - Bocha Web search: Title, URL, Description, Published date and Site name
  - Bocha AI search: Title, URL, Description, Published date, Site name, and structured data card

  If the API key is missing or invalid, appropriate error messages will be returned.
  """
)


@mcp.tool()
async def bocha_web_search(
  query: str,
  freshness: str = "noLimit",
  count: int = 10
) -> str:
  """
  Search with Bocha Web Search and get enhanced search details from billions of web documents,
  including page titles, urls, summaries, site names, site icons, publication dates, image links, and more.

  Args:
    query: Search query (required)
    freshness: The time range for the search results. (Available options YYYY-MM-DD, YYYY-MM-DD..YYYY-MM-DD, noLimit, oneYear, oneMonth, oneWeek, oneDay. Default is noLimit)
    count: Number of results (1-50, default 10)
  """
  
  
  
@mcp.tool()
async def bocha_ai_search(
  query: str,
  freshness: str = "noLimit",
  count: int = 10
) -> str:
  """
  Search with Bocha AI Search, recognizes the semantics of search terms
  and additionally returns structured modal cards with content from vertical domains.

  Args:
    query: Search query (required)
    freshness: The time range for the search results. (Available options noLimit, oneYear, oneMonth, oneWeek, oneDay. Default is noLimit)
    count: Number of results (1-50, default 10)
  """
  
  

def main():
  """Initialize and run the MCP server"""
  

if __name__ == "__name__":
  main()