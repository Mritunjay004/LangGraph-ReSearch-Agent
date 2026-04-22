import { DuckDuckGoSearch } from "@langchain/community/tools/duckduckgo_search";

// Free web search — no API key required (duck-duck-scrape peer dep).
// For higher reliability, swap to Tavily:
//   npm install @langchain/community  (already installed)
//   import { TavilySearchResults } from "@langchain/community/retrievers/tavily_search_api";
//   export const webSearch = new TavilySearchResults({ maxResults: 5 });
//   Add TAVILY_API_KEY=tvly-... to .env (free tier: https://tavily.com)
export const webSearch = new DuckDuckGoSearch({ maxResults: 5 });
