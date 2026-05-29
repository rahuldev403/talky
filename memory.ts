import { Chroma } from "@langchain/community/vectorstores/chroma";
import { OllamaEmbeddings } from "@langchain/ollama";
import { Document } from "@langchain/core/documents";

// 1. Initialize the embedding engine using a local embedding model
// NOTE: Make sure you run `ollama pull nomic-embed-text` in your terminal first!
const embeddings = new OllamaEmbeddings({
  model: "nomic-embed-text",
});

// 2. Connect to the Chroma instance running locally via your Docker Compose file on port 8000
const vectorStore = new Chroma(embeddings, {
  collectionName: "agent_code_memory",
  url: "http://localhost:8000",
});

/**
 * Saves a successfully executed code snippet to the Vector DB.
 */
export async function saveToMemory(
  code: string,
  language: string,
  context: string,
) {
  console.log(`\n[SYSTEM] Saving ${language} snippet to Vector DB...`);
  const doc = new Document({
    pageContent: code,
    metadata: {
      language: language,
      context: context, // e.g., "A python script that prints Hello World"
      timestamp: new Date().toISOString(),
    },
  });

  await vectorStore.addDocuments([doc]);
  console.log(`[SYSTEM] Snippet saved successfully.`);
}

/**
 * Searches the Vector DB for semantically similar code snippets.
 */
export async function searchMemory(query: string) {
  console.log(
    `\n[SYSTEM] Searching Vector DB for context related to: "${query}"...`,
  );
  // Retrieve the top 2 most relevant results
  const results = await vectorStore.similaritySearch(query, 2);

  if (results.length === 0) return "No relevant history found.";

  return results
    .map(
      (doc) =>
        `Context: ${doc.metadata.context}\nLanguage: ${doc.metadata.language}\nCode:\n${doc.pageContent}`,
    )
    .join("\n\n---\n\n");
}
