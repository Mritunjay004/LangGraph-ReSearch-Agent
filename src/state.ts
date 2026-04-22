import "dotenv/config";
import { StateSchema, MessagesValue } from "@langchain/langgraph";
import { z } from "zod";

export const ResearchState = new StateSchema({
  messages: MessagesValue, // append-only LLM turn history

  // ── Pipeline data ──────────────────────────────────────────────
  topic:    z.string().default(""),           // original user topic
  queries:  z.array(z.string()).default([]),  // planner output: search queries
  rawResults: z.array(z.string()).default([]),// researcher output: Tavily JSON strings
  synthesis:  z.string().default(""),         // synthesizer output: key findings
  report:     z.string().default(""),         // writer output: full markdown

  // ── Control flow ───────────────────────────────────────────────
  humanFeedback:   z.string().nullable().default(null), // "approve" or free-text request
  iterationCount:  z.number().default(0),               // loop guard — capped at 2
  reportPath:      z.string().default(""),               // path of saved .md file
});

export type ResearchStateType = typeof ResearchState.State;
