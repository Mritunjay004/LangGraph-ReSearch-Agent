import "dotenv/config";
import * as readline from "node:readline/promises";
import { Command } from "@langchain/langgraph";
import { runStream } from "./stream.js";

const DIVIDER = "─".repeat(60);

async function main() {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  console.log("\n" + DIVIDER);
  console.log("  ReSearch Agent — Autonomous Research Pipeline");
  console.log("  Powered by LangGraph + OpenAI + DuckDuckGo");
  console.log(DIVIDER + "\n");

  const topic = (await rl.question("Research topic: ")).trim();
  if (!topic) {
    console.error("No topic provided. Exiting.");
    rl.close();
    process.exit(1);
  }

  // One thread_id per session — MemorySaver uses this to persist state
  // between the initial run and the interrupt/resume cycle.
  const threadId = crypto.randomUUID();
  const config = { configurable: { thread_id: threadId } };

  console.log("\n" + DIVIDER);
  console.log("Starting research...");
  console.log(DIVIDER);

  // ── Phase 1: Run until interrupt (humanReview node) ────────────────────────
  let result = await runStream({ topic, messages: [] }, config);

  if (!result.interrupted) {
    // Graph completed without interruption (e.g., writer ran directly)
    console.log("\n" + DIVIDER);
    console.log("Research complete.");
    if (result.state.reportPath) console.log(`Report saved: ${result.state.reportPath}`);
    rl.close();
    return;
  }

  // ── Human review ───────────────────────────────────────────────────────────
  // interruptValue holds the payload passed to interrupt() in humanReviewNode
  const synthesis = (result.interruptValue?.synthesis as string) ?? "(no synthesis)";

  console.log("\n\n" + DIVIDER);
  console.log("RESEARCH FINDINGS — REVIEW REQUIRED");
  console.log(DIVIDER);
  console.log(synthesis);
  console.log("\n" + DIVIDER);
  console.log('Type "approve" to generate the full report.');
  console.log("Or describe what additional research you need:");
  console.log(DIVIDER);

  let feedback = "approve";
  try {
    feedback = (await rl.question("\nYour decision: ")).trim() || "approve";
  } catch {
    // Non-TTY mode (piped/redirected input): stdin closed before second read
    console.log("(no input detected — defaulting to approve)");
  }

  console.log("\n" + DIVIDER);
  if (feedback === "approve") {
    console.log("Approved — generating full report...");
  } else {
    console.log(`Requesting additional research: "${feedback}"`);
  }
  console.log(DIVIDER);

  // ── Phase 2: Resume graph with human feedback ──────────────────────────────
  result = await runStream(new Command({ resume: feedback }), config);

  // ── Done ───────────────────────────────────────────────────────────────────
  console.log("\n\n" + DIVIDER);
  console.log("Research complete!");
  if (result.state.reportPath) {
    console.log(`Report saved: ./${result.state.reportPath}`);
  }
  console.log(DIVIDER + "\n");

  rl.close();
}

main().catch((err) => {
  console.error("\nError:", err.message ?? err);
  process.exit(1);
});
