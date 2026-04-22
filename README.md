# ReSearch Agent

A **multi-agent autonomous research pipeline** built with LangGraph, TypeScript, and OpenAI. Enter any topic and the system plans, searches, synthesizes, and writes a structured markdown report — with a human-in-the-loop review step before the final output.

## Architecture

```
User (topic)
     │
     ▼
 [planner]        Breaks topic into 3-4 targeted search queries
     │
     ▼
 [researcher]     Runs all queries via Tavily web search API
     │
     ▼
 [synthesizer]    Extracts 5-7 key findings from raw results
     │
     ▼
 [humanReview]◄──────────────────────────────────────────────┐
     │  interrupt()                                           │
     │  "approve"                     free-text feedback      │
     ▼                                         │              │
 [writer]                              [researcher] ──► [synthesizer]
     │
     ▼
 ./reports/<slug>-<date>.md
```

## LangGraph Patterns Demonstrated

| Pattern | File |
|---|---|
| `StateSchema` with 9 Zod-typed fields | `src/state.ts` |
| `MessagesValue` append reducer | `src/state.ts` |
| 5 specialized agent nodes | `src/agents.ts` |
| Tool use — DuckDuckGo web search (no API key) | `src/tools.ts` |
| `MemorySaver` checkpointer + `thread_id` | `src/graph.ts` |
| Conditional edge routing | `src/graph.ts` |
| `interrupt()` + `Command` resume | `src/agents.ts` |
| Selective streaming (`streamMode: "messages"`) | `src/stream.ts` |
| LangSmith tracing (auto via env vars) | `.env` |

## Prerequisites

- Node.js 18+
- [OpenAI API key](https://platform.openai.com/api-keys)

> **Note on web search**: The project uses DuckDuckGo (no API key required). DuckDuckGo may occasionally rate-limit automated requests. For higher-reliability search, swap `src/tools.ts` to use [Tavily](https://tavily.com) (free tier available) — see the comment in that file.

## Setup

```bash
# 1. Clone and install
git clone <your-repo-url>
cd research-agent
npm install

# 2. Configure environment
cp .env.example .env
# Add your OPENAI_API_KEY

# 3. Run
npm run research
```

## Environment Variables

```env
# Required
OPENAI_API_KEY=sk-...

# Optional: LangSmith tracing (highly recommended — shows full agent traces)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=research-agent
```

## Usage

```
npm run research
```

```
────────────────────────────────────────────────────────────
  ReSearch Agent — Autonomous Research Pipeline
  Powered by LangGraph + OpenAI + Tavily
────────────────────────────────────────────────────────────

Research topic: The impact of transformer architecture on modern AI

[Planning research queries...]
  Queries:
    1. transformer architecture AI history impact
    2. attention mechanism deep learning performance
    3. BERT GPT transformer applications 2024
    4. transformer vs RNN LSTM comparison

[Searching the web...]
[Synthesizing findings...]

────────────────────────────────────────────────────────────
RESEARCH FINDINGS — REVIEW REQUIRED
────────────────────────────────────────────────────────────
1. Transformers replaced RNNs as the dominant sequence model...
...

Type "approve" to generate the full report.
Or describe what additional research you need:

Your decision: approve

[Writing report...]

# The Impact of Transformer Architecture on Modern AI
...

Report saved: ./reports/the-impact-of-transformer-architecture-2025-04-23.md
```

## Also included: LangGraph Tutorial

```bash
npm run tutorial
```

A standalone walkthrough of all core LangGraph concepts in one file (`examples/tutorial.ts`):
- StateGraph + StateSchema
- MessagesValue reducer
- Tool use + ToolNode
- Conditional edges
- MemorySaver + thread_id
- Streaming
- Human-in-the-loop

## Project Structure

```
├── src/
│   ├── state.ts       ResearchState schema (9 Zod fields)
│   ├── tools.ts       Tavily search tool
│   ├── agents.ts      5 agent node functions
│   ├── graph.ts       StateGraph wiring + compile
│   ├── stream.ts      Streaming helper with node filtering
│   └── main.ts        CLI entry point
├── examples/
│   └── tutorial.ts    LangGraph concepts walkthrough
├── reports/           Generated markdown reports (git-ignored)
├── .env.example
└── README.md
```

## Tech Stack

| Package | Version | Purpose |
|---|---|---|
| `@langchain/langgraph` | ^1.2 | Agent graph orchestration |
| `@langchain/openai` | ^1.4 | GPT-4o-mini inference |
| `@langchain/community` | ^1.1 | Tavily search integration |
| `@langchain/core` | ^1.1 | Base LLM abstractions |
| `zod` | ^4 | Runtime schema validation |
| `typescript` | ^6 | Type safety |
