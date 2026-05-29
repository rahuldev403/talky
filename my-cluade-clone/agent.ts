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
import * as readline from "readline";

import { saveToMemory, searchMemory } from "./memory";
import { mapDependency, analyzeImpact } from "./graph";
import { sql, verifyDatabaseConnection } from "./db";

const execAsync = promisify(exec);

// ============================================================================
// UI: Enhanced Step Tracker & Spinner
// ============================================================================
let spinnerInterval: NodeJS.Timeout;
let currentStepNumber = 0;
let executionPlan: string[] = [];
const MAX_STEP_MS = 60_000;
const MAX_RUN_MS = 10 * 60_000;
let runStartedAt = 0;

function initializePlan(steps: string[]) {
  executionPlan = steps;
  currentStepNumber = 0;
  console.log(`\n\x1b[35m📋 SYSTEM EXECUTION ROADMAP:\x1b[0m`);
  steps.forEach((step, index) => {
    console.log(`  \x1b[90m${index + 1}. [ ] ${step}\x1b[0m`);
  });
  console.log("");
}

function advanceToStep(stepIndex: number, detailMessage: string) {
  stopSpinner();
  currentStepNumber = stepIndex;

  if (stepIndex > 0) {
    console.log(`\x1b[32m✔ Step ${stepIndex}: Completed.\x1b[0m`);
  }
  if (executionPlan[stepIndex]) {
    console.log(
      `\x1b[34m🚀 Step ${stepIndex + 1}: ${executionPlan[stepIndex]}\x1b[0m`,
    );
  }

  startSpinner(detailMessage);
}

function extractPlanSteps(text: string): string[] {
  if (!text) return [];
  const lines = text.split(/\r?\n/);
  const steps = lines
    .map((line) => {
      const match = line.match(/^\s*\d+[.)]\s+(.*)$/);
      return match ? match[1].trim() : "";
    })
    .filter((step) => step.length > 0);
  return steps;
}

function startSpinner(message: string) {
  const frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
  let i = 0;
  process.stdout.write("\x1B[?25l"); // Hide cursor
  spinnerInterval = setInterval(() => {
    process.stdout.write(
      `\r\x1b[36m${frames[i]} [Step ${currentStepNumber + 1}] ${message}\x1b[0m`,
    );
    i = (i + 1) % frames.length;
  }, 80);
}

function stopSpinner(successMessage?: string) {
  if (!spinnerInterval) return;
  clearInterval(spinnerInterval);
  process.stdout.write("\r\x1b[K"); // Clear line
  process.stdout.write("\x1B[?25h"); // Show cursor
  if (successMessage) console.log(`\x1b[32m✔ ${successMessage}\x1b[0m`);
}

// ============================================================================
// 1. State Definition
// ============================================================================
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

// ============================================================================
// 2. Define Tools
// ============================================================================

const writeFileTool = tool(
  async ({ filePath, content }) => {
    stopSpinner();
    startSpinner(`Writing file: ${filePath}...`);
    try {
      const dir = path.dirname(filePath);
      if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
      fs.writeFileSync(filePath, content);
      return `Successfully wrote to ${filePath}`;
    } catch (error: any) {
      return `Failed to write file: ${error.message}`;
    }
  },
  {
    name: "write_file",
    description: "Creates or overwrites a file on the local machine.",
    schema: z.object({ filePath: z.string(), content: z.string() }),
  },
);

const readFileTool = tool(
  async ({ filePath }) => {
    stopSpinner();
    startSpinner(`Reading file: ${filePath}...`);
    try {
      return fs.readFileSync(filePath, "utf-8");
    } catch (error: any) {
      return `Failed to read file: ${error.message}`;
    }
  },
  {
    name: "read_file",
    description: "Reads a file from the local machine.",
    schema: z.object({ filePath: z.string() }),
  },
);

const executeCodeTool = tool(
  async ({ code, language }) => {
    stopSpinner();
    startSpinner(`Testing ${language} in Docker...`);
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
      await saveToMemory(code, language, "Execution test");
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
    schema: z.object({ code: z.string(), language: z.string() }),
  },
);

const searchMemoryTool = tool(
  async ({ query }) => {
    stopSpinner();
    startSpinner(`Searching Vector DB...`);
    return await searchMemory(query);
  },
  {
    name: "search_memory",
    description:
      "Searches the vector database for past successful code snippets.",
    schema: z.object({ query: z.string() }),
  },
);

const mapArchitectureTool = tool(
  async ({ sourceFile, targetFile }) => {
    stopSpinner();
    startSpinner(`Graphing dependency...`);
    return await mapDependency(sourceFile, targetFile);
  },
  {
    name: "map_architecture",
    description: "Records a dependency between two files.",
    schema: z.object({ sourceFile: z.string(), targetFile: z.string() }),
  },
);

const analyzeImpactTool = tool(
  async ({ fileName }) => {
    stopSpinner();
    startSpinner(`Analyzing graph impact...`);
    return await analyzeImpact(fileName);
  },
  {
    name: "analyze_impact",
    description: "Checks what other files rely on a specific file.",
    schema: z.object({ fileName: z.string() }),
  },
);

const executeSqlTool = tool(
  async ({ query }) => {
    stopSpinner();
    startSpinner(`Executing SQL query...`);
    try {
      const result = await sql.unsafe(query);
      return `Query executed successfully. Result:\n${JSON.stringify(result, null, 2)}`;
    } catch (error: any) {
      return `SQL Error: ${error.message}`;
    }
  },
  {
    name: "execute_sql",
    description:
      "Executes raw SQL queries against the local PostgreSQL database.",
    schema: z.object({ query: z.string() }),
  },
);

// ============================================================================
// 3. Initialize LLM & Custom Nodes
// ============================================================================

const llm = new ChatOllama({ model: "qwen2.5-coder:7b", temperature: 0 });
const tools = [
  executeCodeTool,
  searchMemoryTool,
  mapArchitectureTool,
  analyzeImpactTool,
  writeFileTool,
  readFileTool,
  executeSqlTool,
];
const llmWithTools = llm.bindTools(tools);

async function callModel(state: typeof AgentState.State) {
  stopSpinner();
  startSpinner("Agent is thinking...");
  let currentMessages = [...state.messages];

  if (state.executedTools.length > 0) {
    const executedList = state.executedTools.join(", ");
    currentMessages.push(
      new SystemMessage(
        `SYSTEM NOTICE: You successfully executed: [${executedList}]. Look at the tool output and proceed to the next step, or summarize.`,
      ),
    );
  }

  const response = await llmWithTools.invoke(currentMessages);
  if (executionPlan.length === 0) {
    const responseText = response.content as string;
    const planSteps = extractPlanSteps(responseText || "");
    if (planSteps.length > 0) initializePlan(planSteps);
  }
  return { messages: [response] };
}

function shouldContinue(state: typeof AgentState.State) {
  const lastMessage = state.messages[state.messages.length - 1];
  const content = lastMessage.content as string;

  if (
    "tool_calls" in lastMessage &&
    Array.isArray(lastMessage.tool_calls) &&
    lastMessage.tool_calls.length > 0
  ) {
    const nextTool = lastMessage.tool_calls[0].name;
    return "tools";
  }

  const fallbackTools = [
    "execute_code",
    "search_memory",
    "map_architecture",
    "analyze_impact",
    "write_file",
    "read_file",
    "execute_sql",
  ];
  for (const t of fallbackTools) {
    if (content && content.includes(`"${t}"`)) return "tools";
  }
  return "__end__";
}

// Merged customToolNode containing both fallback parsing and dynamic UI steps
async function customToolNode(state: typeof AgentState.State) {
  const lastMessage = state.messages[state.messages.length - 1];
  const content = lastMessage.content as string;
  const toolMessages: ToolMessage[] = [];
  const newlyExecuted: string[] = [];

  let toolCallsToProcess = [];

  if (
    "tool_calls" in lastMessage &&
    Array.isArray(lastMessage.tool_calls) &&
    lastMessage.tool_calls.length > 0
  ) {
    toolCallsToProcess = lastMessage.tool_calls;
  } else {
    try {
      const cleanJson = content.substring(
        content.indexOf("{"),
        content.lastIndexOf("}") + 1,
      );
      const parsed = JSON.parse(cleanJson);
      toolCallsToProcess = [
        {
          name: parsed.name || parsed.action,
          args: parsed.arguments || parsed.args || parsed,
          id: "manual_id",
        },
      ];
    } catch (e) {
      /* Ignore text failures */
    }
  }

  for (const toolCall of toolCallsToProcess) {
    let result = "";
    newlyExecuted.push(toolCall.name);

    if (runStartedAt > 0 && Date.now() - runStartedAt > MAX_RUN_MS) {
      stopSpinner();
      result = "Execution stopped: overall time limit reached.";
      toolMessages.push(
        new ToolMessage({
          content: result,
          tool_call_id: toolCall.id || "manual_id",
        }),
      );
      return { messages: toolMessages, executedTools: newlyExecuted };
    }

    if (executionPlan.length > 0 && currentStepNumber < executionPlan.length) {
      advanceToStep(currentStepNumber, `Executing ${toolCall.name}...`);
      currentStepNumber += 1;
    }

    if (toolCall.name === "write_file") {
      const targetFile = toolCall.args.filePath;
      stopSpinner();
      console.log(
        `\x1b[33m⚡ [BUILDING] Writing file system asset: ${targetFile}\x1b[0m`,
      );
      startSpinner(`Writing contents to ${targetFile}...`);
      result = await Promise.race([
        writeFileTool.invoke(toolCall.args),
        new Promise<string>((resolve) =>
          setTimeout(() => resolve("Write timed out after 60s."), MAX_STEP_MS),
        ),
      ]);
    } else if (toolCall.name === "execute_code") {
      stopSpinner();
      console.log(
        `\x1b[33m⚡ [SANDBOX] Running code verification loop...\x1b[0m`,
      );
      startSpinner(`Testing environment inside isolated Docker container...`);
      result = await Promise.race([
        executeCodeTool.invoke(toolCall.args),
        new Promise<string>((resolve) =>
          setTimeout(
            () => resolve("Execution timed out after 60s."),
            MAX_STEP_MS,
          ),
        ),
      ]);
    } else if (toolCall.name === "execute_sql") {
      stopSpinner();
      console.log(
        `\x1b[33m⚡ [DATABASE] Modifying local database schemas...\x1b[0m`,
      );
      startSpinner(`Running query against claudedb...`);
      result = await Promise.race([
        executeSqlTool.invoke(toolCall.args),
        new Promise<string>((resolve) =>
          setTimeout(() => resolve("SQL timed out after 60s."), MAX_STEP_MS),
        ),
      ]);
    } else {
      const matchedTool = tools.find((t) => t.name === toolCall.name);
      if (matchedTool) {
        result = await Promise.race([
          matchedTool.invoke(toolCall.args),
          new Promise<string>((resolve) =>
            setTimeout(() => resolve("Tool timed out after 60s."), MAX_STEP_MS),
          ),
        ]);
      } else {
        result = "Unknown tool requested.";
      }
    }

    toolMessages.push(
      new ToolMessage({
        content: result,
        tool_call_id: toolCall.id || "manual_id",
      }),
    );
  }

  return {
    messages:
      toolMessages.length > 0
        ? toolMessages
        : [
            new ToolMessage({
              content: "Error parsing tools",
              tool_call_id: "manual_id",
            }),
          ],
    executedTools: newlyExecuted,
  };
}

// ============================================================================
// 4. Build Graph
// ============================================================================

const workflow = new StateGraph(AgentState)
  .addNode("agent", callModel)
  .addNode("tools", customToolNode)
  .addEdge("__start__", "agent")
  .addConditionalEdges("agent", shouldContinue)
  .addEdge("tools", "agent");

const app = workflow.compile();

// ============================================================================
// 5. Interactive CLI
// ============================================================================

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const MASTER_INSTRUCTION = new SystemMessage(
  `You are an elite autonomous AI developer. 
    CRITICAL WORKFLOW RULES:
    1. When handled a massive task (like building an app), your first response should outline the files you intend to make.
    2. You MUST use the 'write_file' tool to write out the physical files onto the machine. Do not just print them in text.
    3. You MUST use 'execute_code' to run docker instances (e.g., pulling and spinning up MongoDB images).
    4. Proceed through your work step-by-step, explaining briefly which file or component you are wiring next before invoking the tool.`,
);

async function runInteractiveCLI() {
  console.clear();
  console.log(
    "\x1b[36m==================================================\x1b[0m",
  );
  console.log(
    "\x1b[1m🤖 Terminal Agent Initialized (Type 'exit' to quit)\x1b[0m",
  );
  console.log(
    "\x1b[36m==================================================\x1b[0m\n",
  );

  startSpinner("Verifying database connections...");
  const isDbConnected = await verifyDatabaseConnection();
  stopSpinner();

  if (!isDbConnected) {
    console.log(
      "\x1b[31m⚠️ Warning: Database is offline. SQL tools will fail until Docker is running.\x1b[0m\n",
    );
  } else {
    console.log(
      "\x1b[32m✓ PostgreSQL Database connected successfully.\x1b[0m\n",
    );
  }

  let currentConfig = { configurable: { thread_id: "1" } };
  let currentState = { messages: [MASTER_INSTRUCTION], executedTools: [] };
  runStartedAt = Date.now();

  const askQuestion = () => {
    rl.question("\x1b[32m❯ You: \x1b[0m", async (userInput) => {
      if (userInput.toLowerCase() === "exit") {
        stopSpinner();
        rl.close();
        process.exit(0);
      }

      currentState.messages.push(new HumanMessage(userInput));
      currentState.executedTools = [];

      try {
        const finalState = await app.invoke(currentState, {
          ...currentConfig,
          recursionLimit: 150,
        });
        stopSpinner();
        currentState = finalState;
        const lastMessage = finalState.messages[finalState.messages.length - 1];
        console.log(
          `\n\x1b[1m🤖 Claude Clone:\x1b[0m\n${lastMessage.content}\n`,
        );
      } catch (error) {
        stopSpinner();
        console.error("\n[SYSTEM ERROR] Graph execution failed:", error);
      }
      askQuestion();
    });
  };
  askQuestion();
}

runInteractiveCLI();
