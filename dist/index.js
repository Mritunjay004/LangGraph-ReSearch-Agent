"use strict";
/**
 * LangGraph Complete Tour — covers every core concept in one file.
 *
 * Concepts covered (in order):
 *   1. Tool definition with Zod schemas
 *   2. StateGraph + StateSchema (Zod-based state)
 *   3. MessagesValue — append reducer (never overwrites)
 *   4. Nodes — pure async functions (state-in → patch-out)
 *   5. ToolNode — automatic tool dispatch from tool_calls
 *   6. Conditional edges — route to tools or END
 *   7. MemorySaver checkpointer — persist state by thread_id
 *   8. Multi-turn conversations — same thread_id carries history
 *   9. Streaming — tokens arrive as they're generated
 *  10. Human-in-the-loop — interrupt() pauses for human input, Command resumes
 *
 * Run:  npx tsc && node dist/index.js
 * Env:  OPENAI_API_KEY in .env
 */
Object.defineProperty(exports, "__esModule", { value: true });
require("dotenv/config");
const openai_1 = require("@langchain/openai");
const tools_1 = require("@langchain/core/tools");
const messages_1 = require("@langchain/core/messages");
const langgraph_1 = require("@langchain/langgraph");
const prebuilt_1 = require("@langchain/langgraph/prebuilt");
const zod_1 = require("zod");
// ══════════════════════════════════════════════════════
// 1. TOOLS — Zod schemas give the LLM typed argument specs
// ══════════════════════════════════════════════════════
const calculator = (0, tools_1.tool)(async ({ a, b, op }) => {
    if (op === "add")
        return String(a + b);
    if (op === "sub")
        return String(a - b);
    if (op === "mul")
        return String(a * b);
    if (op === "div")
        return b === 0 ? "Error: division by zero" : String(a / b);
    return "Unknown operation";
}, {
    name: "calculator",
    description: "Perform basic arithmetic: add, sub, mul, div.",
    schema: zod_1.z.object({
        a: zod_1.z.number().describe("First operand"),
        b: zod_1.z.number().describe("Second operand"),
        op: zod_1.z.enum(["add", "sub", "mul", "div"]).describe("Operation"),
    }),
});
const getWeather = (0, tools_1.tool)(async ({ city }) => {
    // Stubbed — swap with a real API call (e.g. OpenWeatherMap)
    const data = {
        London: "12°C, cloudy",
        Tokyo: "28°C, sunny",
        NYC: "18°C, partly cloudy",
    };
    return data[city] ?? `No weather data for "${city}"`;
}, {
    name: "get_weather",
    description: "Get current weather for a city.",
    schema: zod_1.z.object({
        city: zod_1.z.string().describe("City name, e.g. 'Tokyo'"),
    }),
});
const tools = [calculator, getWeather];
// ══════════════════════════════════════════════════════
// 2. STATE — one schema, shared across all graph nodes
//
// MessagesValue is a special list reducer: each node returns
// NEW messages to append; the graph never replaces the array.
// Any other Zod field uses last-write-wins by default.
// ══════════════════════════════════════════════════════
const AgentState = new langgraph_1.StateSchema({
    messages: langgraph_1.MessagesValue,
});
// ══════════════════════════════════════════════════════
// 3. MODEL — bind tools so the LLM can emit `tool_calls`
// ══════════════════════════════════════════════════════
const llm = new openai_1.ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });
const llmWithTools = llm.bindTools(tools);
// ══════════════════════════════════════════════════════
// 4. NODES — receive full state, return only what changed
// ══════════════════════════════════════════════════════
// Calls the LLM; if the LLM wants a tool, the response includes tool_calls.
const agentNode = async (state) => {
    const response = await llmWithTools.invoke(state.messages);
    return { messages: [response] };
};
// ToolNode reads tool_calls from the last AIMessage, runs the
// matching tool, and appends the ToolMessage result automatically.
const toolNode = new prebuilt_1.ToolNode(tools);
// ══════════════════════════════════════════════════════
// 5. CONDITIONAL EDGE — decide what happens after the LLM
// ══════════════════════════════════════════════════════
const shouldContinue = (state) => {
    const last = state.messages.at(-1);
    // Only AIMessages carry tool_calls; guard against ToolMessages here
    if (!last || !messages_1.AIMessage.isInstance(last))
        return langgraph_1.END;
    return last.tool_calls?.length ? "tools" : langgraph_1.END;
};
// ══════════════════════════════════════════════════════
// 6. GRAPH — wire nodes + edges, then compile
//
// Flow:  START → agent → (tool_calls?) → tools → agent → … → END
//
// MemorySaver stores a snapshot after every node so the graph
// can be resumed by thread_id across separate .invoke() calls.
// ══════════════════════════════════════════════════════
const checkpointer = new langgraph_1.MemorySaver();
const agentGraph = new langgraph_1.StateGraph(AgentState)
    .addNode("agent", agentNode)
    .addNode("tools", toolNode)
    .addEdge(langgraph_1.START, "agent")
    // conditional edge: "tools" or END
    .addConditionalEdges("agent", shouldContinue, ["tools", langgraph_1.END])
    // after every tool call, give the LLM a chance to respond or call another tool
    .addEdge("tools", "agent")
    .compile({ checkpointer });
// ══════════════════════════════════════════════════════
// 7. HUMAN-IN-THE-LOOP sub-graph
//
// interrupt() suspends the graph mid-node and surfaces a
// payload to the caller.  The caller resumes with
// Command({ resume: <value> }) on the same thread_id.
// ══════════════════════════════════════════════════════
const HilState = new langgraph_1.StateSchema({
    task: zod_1.z.string(),
    // nullable so we can start without a decision
    approved: zod_1.z.boolean().nullable(),
    result: zod_1.z.string().nullable(),
});
const hilGraph = new langgraph_1.StateGraph(HilState)
    .addNode("request_approval", async (state) => {
    // Execution pauses here; the caller sees __interrupt__ in the returned state
    const decision = (0, langgraph_1.interrupt)({
        question: `Approve task: "${state.task}"?`,
        // Use strings, not booleans — LangGraph treats `false` as empty input
        hint: 'Resume with Command({ resume: "approve" }) or Command({ resume: "reject" })',
    });
    return new langgraph_1.Command({ goto: decision === "approve" ? "execute" : "reject" });
}, { ends: ["execute", "reject"] })
    .addNode("execute", (state) => ({
    approved: true,
    result: `✓ Executed: ${state.task}`,
}))
    .addNode("reject", () => ({
    approved: false,
    result: "✗ Task rejected by human.",
}))
    .addEdge(langgraph_1.START, "request_approval")
    .addEdge("execute", langgraph_1.END)
    .addEdge("reject", langgraph_1.END)
    .compile({ checkpointer: new langgraph_1.MemorySaver() });
// ══════════════════════════════════════════════════════
// 8. DEMOS
// ══════════════════════════════════════════════════════
async function main() {
    // ── Demo 1: Tool use (calculator) ─────────────────
    console.log("\n═══ Demo 1: Tool use ═══");
    const r1 = await agentGraph.invoke({ messages: [{ role: "user", content: "What is (42 × 7) + 18?" }] }, { configurable: { thread_id: "d1" } });
    console.log("Answer:", r1.messages.at(-1)?.content);
    // ── Demo 2: Multi-turn with persistent thread ──────
    // Each call with the same thread_id appends to existing history;
    // the LLM sees all prior turns automatically via the checkpointer.
    console.log("\n═══ Demo 2: Multi-turn conversation ═══");
    const cfg = { configurable: { thread_id: "chat-1" } };
    const r2a = await agentGraph.invoke({ messages: [{ role: "user", content: "Weather in Tokyo?" }] }, cfg);
    console.log("Turn 1:", r2a.messages.at(-1)?.content);
    // "And in London?" — the LLM knows we're still talking about weather
    const r2b = await agentGraph.invoke({ messages: [{ role: "user", content: "And in London?" }] }, cfg);
    console.log("Turn 2:", r2b.messages.at(-1)?.content);
    // ── Demo 3: Streaming ──────────────────────────────
    // streamMode "messages" yields [chunk, metadata] tuples so you
    // can pipe tokens to a UI in real time.
    console.log("\n═══ Demo 3: Streaming ═══");
    process.stdout.write("Stream > ");
    const stream = await agentGraph.stream({
        messages: [
            {
                role: "user",
                content: "Explain what a LangGraph StateGraph is in one sentence.",
            },
        ],
    }, { configurable: { thread_id: "d3" }, streamMode: "messages" });
    for await (const [chunk] of stream) {
        if (typeof chunk.content === "string" && chunk.content) {
            process.stdout.write(chunk.content);
        }
    }
    console.log("\n");
    // ── Demo 4: Human-in-the-loop ─────────────────────
    console.log("═══ Demo 4: Human-in-the-loop ═══");
    // Approved path
    const hilCfg1 = { configurable: { thread_id: "hil-approve" } };
    const paused = await hilGraph.invoke({ task: "Deploy v2.1 to production", approved: null, result: null }, hilCfg1);
    // Graph is now paused; __interrupt__ contains the payload from interrupt()
    console.log("Paused — interrupt payload:", paused.__interrupt__?.[0]
        ?.value);
    // Human says yes → "approve" routes to "execute"
    const approved = await hilGraph.invoke(new langgraph_1.Command({ resume: "approve" }), hilCfg1);
    console.log("Approved result:", approved.result);
    // Rejected path
    const hilCfg2 = { configurable: { thread_id: "hil-reject" } };
    await hilGraph.invoke({ task: "Delete all production logs", approved: null, result: null }, hilCfg2);
    // Human says no → "reject" routes to "reject" node
    const rejected = await hilGraph.invoke(new langgraph_1.Command({ resume: "reject" }), hilCfg2);
    console.log("Rejected result:", rejected.result);
}
main().catch(console.error);
