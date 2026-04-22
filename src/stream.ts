import type { RunnableConfig } from "@langchain/core/runnables";
import { researchGraph } from "./graph.js";
import type { ResearchStateType } from "./state.js";

// Node labels shown in the terminal while the graph is running
const NODE_LABELS: Record<string, string> = {
  planner:     "Planning research queries",
  researcher:  "Searching the web",
  synthesizer: "Synthesizing findings",
  humanReview: "Preparing your review",
  writer:      "Writing report",
};

export interface StreamResult {
  state: ResearchStateType;
  interrupted: boolean;
  interruptValue?: Record<string, unknown>;
}

// Streams the graph and returns the final state + interrupt info.
// Writer tokens are streamed to stdout in real time; other nodes print a label.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function runStream(input: any, config: RunnableConfig): Promise<StreamResult> {
  const stream = await researchGraph.stream(input, {
    ...config,
    streamMode: "messages",
  });

  let lastNode = "";

  for await (const [chunk, metadata] of stream as AsyncIterable<[any, any]>) {
    const node: string = metadata?.langgraph_node ?? "";

    // Print a progress label once per node transition
    if (node && node !== lastNode) {
      if (node !== "writer") {
        process.stdout.write(`\n[${NODE_LABELS[node] ?? node}...]\n`);
      } else {
        process.stdout.write(`\n[Writing report...]\n\n`);
      }
      lastNode = node;
    }

    // Stream writer tokens directly — all other node output is suppressed
    if (node === "writer" && typeof chunk.content === "string" && chunk.content) {
      process.stdout.write(chunk.content);
    }
  }

  // After stream ends, inspect the full state snapshot.
  // When interrupt() fires, the graph pauses and stores the payload in
  // snapshot.tasks[0].interrupts — NOT in snapshot.values.__interrupt__.
  const snapshot = await researchGraph.getState(config);
  const tasks = (snapshot.tasks ?? []) as Array<{ interrupts?: Array<{ value: unknown }> }>;
  const firstInterrupt = tasks.find((t) => t.interrupts?.length)?.interrupts?.[0];

  return {
    state: snapshot.values as ResearchStateType,
    interrupted: !!firstInterrupt,
    interruptValue: firstInterrupt?.value as Record<string, unknown> | undefined,
  };
}
