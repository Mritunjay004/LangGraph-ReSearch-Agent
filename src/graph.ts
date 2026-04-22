import { StateGraph, MemorySaver, START, END } from "@langchain/langgraph";
import { ResearchState } from "./state.js";
import {
  plannerNode,
  researcherNode,
  synthesizerNode,
  humanReviewNode,
  writerNode,
} from "./agents.js";
import type { ResearchStateType } from "./state.js";

// ── Conditional edge ───────────────────────────────────────────────────────────
// Called after humanReview. Routes to writer on approval, or loops back to
// researcher for additional searches (capped at 2 iterations to prevent runaway).
function routeAfterReview(state: ResearchStateType): "writer" | "researcher" {
  if (state.humanFeedback === "approve") return "writer";
  if (state.iterationCount < 2) return "researcher";
  return "writer"; // safety valve — force completion after 2 iterations
}

// ── Graph ──────────────────────────────────────────────────────────────────────
// MemorySaver persists the full state snapshot after every node, keyed by
// thread_id. This is what makes interrupt() + Command({ resume }) work.
const checkpointer = new MemorySaver();

export const researchGraph = new StateGraph(ResearchState)
  .addNode("planner",     plannerNode)
  .addNode("researcher",  researcherNode)
  .addNode("synthesizer", synthesizerNode)
  .addNode("humanReview", humanReviewNode)
  .addNode("writer",      writerNode)

  // Linear pipeline
  .addEdge(START,        "planner")
  .addEdge("planner",    "researcher")
  .addEdge("researcher", "synthesizer")
  .addEdge("synthesizer","humanReview")

  // Conditional: approve → writer, feedback → researcher (loop)
  .addConditionalEdges("humanReview", routeAfterReview, ["writer", "researcher"])

  .addEdge("writer", END)
  .compile({ checkpointer });
