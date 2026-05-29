import postgres from "postgres";

// The direct connection string pointing to your Docker container
const DATABASE_URL = "postgres://admin:securepassword@localhost:5432/claudedb";

/**
 * Shared, pooling-safe PostgreSQL client.
 * Using lazy allocation so it only opens connections when a tool actively queries it.
 */
export const sql = postgres(DATABASE_URL, {
  prepare: false, // Disables prefetch which isn't needed for raw tool executions
  idle_timeout: 20, // Automatically closes idle connections after 20 seconds to save system memory
  max: 10, // Limits the maximum connection pool size
});

/**
 * Utility function to verify the database connection is alive before the agent runs
 */
export async function verifyDatabaseConnection(): Promise<boolean> {
  try {
    await sql`SELECT 1`;
    return true;
  } catch (error: any) {
    console.error(
      `\n\x1b[31m[DATABASE ERROR] Failed to connect to PostgreSQL: ${error.message}\x1b[0m`,
    );
    return false;
  }
}
