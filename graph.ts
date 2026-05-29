import neo4j from "neo4j-driver";

// Connect to the Neo4j container running on port 7687
const driver = neo4j.driver(
  "bolt://localhost:7687",
  neo4j.auth.basic("neo4j", "securepassword"),
);

/**
 * Maps a structural dependency between two files in the project.
 */
export async function mapDependency(
  sourceFile: string,
  targetFile: string,
  relationType: string = "IMPORTS",
) {
  console.log(
    `\n[SYSTEM] Graphing architecture: ${sourceFile} -> ${relationType} -> ${targetFile}`,
  );
  const session = driver.session();
  try {
    // Cypher Query Language to merge (create if not exists) nodes and relationships
    await session.run(
      `
            MERGE (a:File {name: $sourceFile})
            MERGE (b:File {name: $targetFile})
            MERGE (a)-[r:DEPENDS_ON {type: $relationType}]->(b)
            `,
      { sourceFile, targetFile, relationType },
    );
    return `Successfully mapped connection: ${sourceFile} depends on ${targetFile}`;
  } catch (error) {
    return `Graph mapping failed: ${error}`;
  } finally {
    await session.close();
  }
}

/**
 * Queries the graph to see what other files might break if we change a target file.
 */
export async function analyzeImpact(fileName: string) {
  console.log(`\n[SYSTEM] Analyzing structural impact for: ${fileName}`);
  const session = driver.session();
  try {
    // Find any file (a) that depends on our target file (b)
    const result = await session.run(
      `MATCH (a:File)-[:DEPENDS_ON]->(b:File {name: $fileName}) RETURN a.name AS dependentFile`,
      { fileName },
    );

    if (result.records.length === 0)
      return `${fileName} has no known dependencies. Safe to modify.`;

    const dependents = result.records.map((record) =>
      record.get("dependentFile"),
    );
    return `WARNING: Modifying ${fileName} may impact the following files: ${dependents.join(", ")}. Review them before proceeding.`;
  } catch (error) {
    return `Impact analysis failed: ${error}`;
  } finally {
    await session.close();
  }
}
