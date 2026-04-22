import { promises as fs } from "node:fs";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { interrupt } from "@langchain/langgraph";
import { webSearch } from "./tools.js";
import type { ResearchStateType } from "./state.js";

const llm = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });
const llmWriter = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0.3 });

// ── Planner ───────────────────────────────────────────────────────────────────
// Breaks the topic into 3-4 targeted search queries.
export async function plannerNode(state: ResearchStateType) {
  const response = await llm.invoke([
    new SystemMessage(
      "You are a research planner. Generate exactly 3-4 precise, diverse web search queries " +
      "that together cover the topic comprehensively. " +
      "Return ONLY a valid JSON array of strings — no other text, no markdown."
    ),
    new HumanMessage(`Research topic: ${state.topic}`),
  ]);

  let queries: string[] = [];
  try {
    queries = JSON.parse(response.content as string);
  } catch {
    // Graceful fallback if the model wraps output in markdown fences
    const match = (response.content as string).match(/\[[\s\S]*\]/);
    if (match) queries = JSON.parse(match[0]);
  }

  console.log(`\n  Queries: ${queries.map((q, i) => `\n    ${i + 1}. ${q}`).join("")}\n`);
  return { messages: [response], queries };
}

// ── Researcher ────────────────────────────────────────────────────────────────
// Calls Tavily once per query. Direct invocation (not ToolNode) because this
// is a pipeline, not a ReAct loop — each query is deterministic.
export async function researcherNode(state: ResearchStateType) {
  const rawResults: string[] = [];

  for (const query of state.queries) {
    console.log(`  Searching: "${query}"`);
    try {
      // tavilySearch.invoke returns a JSON string of result objects
      const result = await webSearch.invoke(query);
      rawResults.push(result);
    } catch (err) {
      rawResults.push(`Search failed for "${query}": ${String(err)}`);
    }
  }

  return {
    rawResults,
    iterationCount: state.iterationCount + 1,
  };
}

// ── Synthesizer ───────────────────────────────────────────────────────────────
// Extracts the most important findings from raw Tavily results.
export async function synthesizerNode(state: ResearchStateType) {
  const combinedResults = state.rawResults
    .map((r, i) => `[Result ${i + 1}]\n${r}`)
    .join("\n\n---\n\n");

  const response = await llm.invoke([
    new SystemMessage(
      "You are a research analyst. Given raw web search results, extract the 5-7 most important, " +
      "concrete findings relevant to the topic. Be factual and concise. " +
      "Format as a numbered list."
    ),
    new HumanMessage(
      `Topic: ${state.topic}\n\nSearch Results:\n${combinedResults}`
    ),
  ]);

  return { messages: [response], synthesis: response.content as string };
}

// ── Human Review ──────────────────────────────────────────────────────────────
// Pauses the graph so the user can review findings and decide to approve or
// request additional research. Routing is handled by the conditional edge in
// graph.ts — this node stays pure (returns only state updates).
export async function humanReviewNode(state: ResearchStateType) {
  // interrupt() suspends execution here. The caller sees __interrupt__ in the
  // returned state. Resuming with Command({ resume: "approve" }) or
  // Command({ resume: "<feedback>" }) continues from this exact line.
  const feedback = interrupt({
    synthesis: state.synthesis,
    instruction:
      'Type "approve" to generate the final report, or describe what additional research you need.',
  }) as string;

  return { humanFeedback: feedback };
}

// ── Writer ────────────────────────────────────────────────────────────────────
// Produces a full structured markdown report and saves it to ./reports/.
export async function writerNode(state: ResearchStateType) {
  const response = await llmWriter.invoke([
    new SystemMessage(
      "You are a professional technical writer. Write a comprehensive, well-structured Markdown " +
      "research report. Include these sections: " +
      "## Executive Summary, ## Key Findings, ## Detailed Analysis, ## Methodology, ## Conclusion. " +
      "Use clear headings, bullet points where appropriate, and cite search results as [Source N]."
    ),
    new HumanMessage(
      `Topic: ${state.topic}\n\n` +
      `Research Queries Used:\n${state.queries.map((q, i) => `${i + 1}. ${q}`).join("\n")}\n\n` +
      `Synthesized Findings:\n${state.synthesis}`
    ),
  ]);

  const report = response.content as string;

  // Save to ./reports/<slug>-<timestamp>.md
  const slug = state.topic
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .slice(0, 50);
  const timestamp = new Date().toISOString().slice(0, 10);
  const reportPath = `reports/${slug}-${timestamp}.md`;

  await fs.mkdir("reports", { recursive: true });
  await fs.writeFile(reportPath, report, "utf-8");

  return { messages: [response], report, reportPath };
}
