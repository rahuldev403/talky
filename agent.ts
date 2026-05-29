import {
  StateGraph,
  Annotation,
  messagesStateReducer,
} from "@langchain/langgraph";
import { ChatOllama } from "@langchain/ollama";
import {
  BaseMessage,
  HumanMessage,
  ToolMessage,
  SystemMessage,
} from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import * as fs from "fs";
import * as path from "path";
import { exec } from "child_process";
import { promisify } from "util";
import { saveToMemory, searchMemory } from "./memory";
import { mapDependency, analyzeImpact } from "./graph";

const execAsync = promisify(exec);

// 1. State Definition
const AgentState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: messagesStateReducer,
    default: () => [],
  }),
  executedTools: Annotation<string[]>({
    reducer: (left, right) => left.concat(right),
    default: () => [],
  }),
});

// ---------------------------------------------------------------------------
// 2. Define Tools
// ---------------------------------------------------------------------------

const executeCodeTool = tool(
  async ({ code, language, objective }) => {
    console.log(`\n[SYSTEM] Spinning up Docker to run ${language} code...`);
    const tmpDir = path.join(process.cwd(), "tmp_sandbox");
    if (!fs.existsSync(tmpDir)) fs.mkdirSync(tmpDir);

    let fileName, dockerImage, runCmd;
    if (language.toLowerCase() === "python") {
      fileName = "script.py";
      dockerImage = "python:3.9-alpine";
      runCmd = "python script.py";
    } else if (
      language.toLowerCase() === "javascript" ||
      language.toLowerCase() === "typescript"
    ) {
      fileName = "script.js";
      dockerImage = "node:18-alpine";
      runCmd = "node script.js";
    } else return `Error: Unsupported language.`;

    const filePath = path.join(tmpDir, fileName);
    fs.writeFileSync(filePath, code);

    try {
      const { stdout, stderr } = await execAsync(
        `docker run --rm --network none -v "${tmpDir}:/app" -w /app ${dockerImage} ${runCmd}`,
        { timeout: 10000 },
      );
      await saveToMemory(code, language, objective || "Execution test");
      return `Execution Output:\n${stdout}\n${stderr ? "Warnings:\n" + stderr : ""}`;
    } catch (error: any) {
      return `Execution Failed:\n${error.message}\n${error.stderr || ""}`;
    } finally {
      if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
    }
  },
  {
    name: "execute_code",
    description: "Executes code inside the secure Docker sandbox.",
    schema: z.object({
      code: z.string(),
      language: z.string(),
      objective: z.string().optional(),
    }),
  },
);

const searchMemoryTool = tool(async ({ query }) => await searchMemory(query), {
  name: "search_memory",
  description:
    "Searches the vector database for past successful code snippets.",
  schema: z.object({ query: z.string() }),
});

const mapArchitectureTool = tool(
  async ({ sourceFile, targetFile }) =>
    await mapDependency(sourceFile, targetFile),
  {
    name: "map_architecture",
    description: "Records a dependency between two files.",
    schema: z.object({ sourceFile: z.string(), targetFile: z.string() }),
  },
);

const analyzeImpactTool = tool(
  async ({ fileName }) => await analyzeImpact(fileName),
  {
    name: "analyze_impact",
    description:
      "Checks what other files rely on a specific file before modifying it.",
    schema: z.object({ fileName: z.string() }),
  },
);

// ---------------------------------------------------------------------------
// 3. Initialize LLM & Custom Nodes
// ---------------------------------------------------------------------------

const llm = new ChatOllama({ model: "qwen2.5-coder:7b", temperature: 0 });
const tools = [
  executeCodeTool,
  searchMemoryTool,
  mapArchitectureTool,
  analyzeImpactTool,
];
const llmWithTools = llm.bindTools(tools);

async function callModel(state: typeof AgentState.State) {
  let currentMessages = [...state.messages];

  // THE FIX: Dynamically tell the model about ALL tools it just ran so it stops asking
  if (state.executedTools.length > 0) {
    const executedList = state.executedTools.join(", ");
    currentMessages.push(
      new SystemMessage(
        `SYSTEM NOTICE: You have already successfully executed the following tools: [${executedList}]. Do NOT call them again. Look at the tool output in the chat history and proceed to the next step, or provide your final response.`,
      ),
    );
  }

  const response = await llmWithTools.invoke(currentMessages);
  return { messages: [response] };
}

// THE FIX: Universal Bulletproof Router
function shouldContinue(state: typeof AgentState.State) {
  const lastMessage = state.messages[state.messages.length - 1];
  const content = lastMessage.content as string;

  // Native API Guard
  if (
    "tool_calls" in lastMessage &&
    Array.isArray(lastMessage.tool_calls) &&
    lastMessage.tool_calls.length > 0
  ) {
    const nextTool = lastMessage.tool_calls[0].name;
    if (state.executedTools.includes(nextTool)) {
      console.log(
        `\n[SYSTEM GUARD] Intercepted duplicate loop for '${nextTool}'. Forcing exit.`,
      );
      return "__end__";
    }
    return "tools";
  }

  // Fallback Text Parser Guard
  const fallbackTools = [
    "execute_code",
    "search_memory",
    "map_architecture",
    "analyze_impact",
  ];
  for (const tool of fallbackTools) {
    if (content && content.includes(`"${tool}"`)) {
      if (state.executedTools.includes(tool)) {
        console.log(
          `\n[SYSTEM GUARD] Intercepted duplicate fallback loop for '${tool}'. Forcing exit.`,
        );
        return "__end__";
      }
      return "tools";
    }
  }

  return "__end__";
}

// BULLETPROOF EXECUTOR
async function customToolNode(state: typeof AgentState.State) {
  const lastMessage = state.messages[state.messages.length - 1];
  const content = lastMessage.content as string;
  const toolMessages: ToolMessage[] = [];
  const newlyExecuted: string[] = [];

  // Case A: Native tool calls
  if (
    "tool_calls" in lastMessage &&
    Array.isArray(lastMessage.tool_calls) &&
    lastMessage.tool_calls.length > 0
  ) {
    for (const toolCall of lastMessage.tool_calls) {
      let result = "";
      newlyExecuted.push(toolCall.name);

      if (toolCall.name === "execute_code")
        result = await executeCodeTool.invoke(toolCall.args);
      else if (toolCall.name === "search_memory")
        result = await searchMemoryTool.invoke(toolCall.args);
      else if (toolCall.name === "map_architecture")
        result = await mapArchitectureTool.invoke(toolCall.args);
      else if (toolCall.name === "analyze_impact")
        result = await analyzeImpactTool.invoke(toolCall.args);

      toolMessages.push(
        new ToolMessage({
          content: result,
          tool_call_id: toolCall.id ?? "manual_id",
        }),
      );
    }
    return { messages: toolMessages, executedTools: newlyExecuted };
  }

  // Case B: Fallback Text JSON Parsing
  try {
    const cleanJson = content.substring(
      content.indexOf("{"),
      content.lastIndexOf("}") + 1,
    );
    const parsed = JSON.parse(cleanJson);
    const toolName = parsed.name || parsed.action;
    const args = parsed.arguments || parsed.args || parsed;

    let result = "";
    newlyExecuted.push(toolName);

    if (toolName === "execute_code")
      result = await executeCodeTool.invoke(args);
    else if (toolName === "search_memory")
      result = await searchMemoryTool.invoke(args);
    else if (toolName === "map_architecture")
      result = await mapArchitectureTool.invoke(args);
    else if (toolName === "analyze_impact")
      result = await analyzeImpactTool.invoke(args);
    else result = "Unknown tool requested.";

    return {
      messages: [
        new ToolMessage({ content: result, tool_call_id: "manual_id" }),
      ],
      executedTools: newlyExecuted,
    };
  } catch (err) {
    return {
      messages: [
        new ToolMessage({
          content: "Error parsing tool parameters from text.",
          tool_call_id: "manual_id",
        }),
      ],
      executedTools: [],
    };
  }
}

// ---------------------------------------------------------------------------
// 4. Build Graph
// ---------------------------------------------------------------------------

const workflow = new StateGraph(AgentState)
  .addNode("agent", callModel)
  .addNode("tools", customToolNode)
  .addEdge("__start__", "agent")
  .addConditionalEdges("agent", shouldContinue)
  .addEdge("tools", "agent");

const app = workflow.compile();

async function runAgent() {
  console.log("Agent is thinking...\n");

  const systemInstruction = new SystemMessage(
    "You are an elite architectural AI. First, use `map_architecture` to record that 'UserAuth.ts' depends on 'DatabaseConnection.ts'. Then, use `analyze_impact` to check what happens if we decide to rewrite 'DatabaseConnection.ts'.",
  );

  const finalState = await app.invoke({
    messages: [systemInstruction],
    executedTools: [],
  });

  const lastMessage = finalState.messages[finalState.messages.length - 1];
  console.log("\n[AGENT RESPONSE]:\n" + lastMessage.content);
}

runAgent();
