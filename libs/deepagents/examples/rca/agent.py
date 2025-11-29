import os
import sys
import json
import argparse
from dotenv import load_dotenv

# Add the project root to sys.path to ensure we can import deepagents if running from here
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from deepagents import create_deep_agent
from examples.rca.tools import list_tables_in_directory, get_schema, query_parquet_files

# Load environment variables (for API keys)
# Explicitly look for .env in the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path)

# Check if keys are valid (not placeholders)
if os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-xxxx":
    print(f"Error: ANTHROPIC_API_KEY in {dotenv_path} is still the default placeholder.")
    print("Please edit the .env file and add your actual API key.")
    sys.exit(1)


# Define the System Prompt for the RCA Agent
RCA_SYSTEM_PROMPT = """
**Mission Context:**
You are investigating a system failure in namespace `ts0`.
**Abnormal Period (Fault Injection):** '2025-07-23 14:10:23' to '2025-07-23 14:14:23' UTC
**Normal Period (Baseline):** '2025-07-23 14:06:23' to '2025-07-23 14:10:23' UTC

**Objective:**
Analyze the span metrics, trace data, and logs in the current directory to identify the **Root Cause Service**.

**Analysis Workflow:**

Step 1: Discover and Understand Data
- Use `list_tables_in_directory` to find available parquet files.
- Use `get_schema` on key files (logs, traces, metrics) to understand columns.
- *Note: Do this once. Do not repeat.*

Step 2: High-Level Problem Overview
- Query `conclusion.parquet` (if available) or summarize the general error patterns.
- Identify the initial symptoms (e.g., which service is reporting errors?).

Step 3: Analyze Anomalous Data (Focus on Abnormal Period)
- Extract errors and high latency events specifically within the **Abnormal Period**.
- **Query Example:**
  ```sql
  SELECT service_name, level, COUNT(*) as count 
  FROM abnormal_logs 
  WHERE time >= TIMESTAMP '2025-07-23 14:10:23' 
    AND time <= TIMESTAMP '2025-07-23 14:14:23'
  GROUP BY service_name, level 
  ORDER BY count DESC 
  LIMIT 50
  ```

Step 4: Compare with Normal Data (Focus on Normal Period)
- Establish a baseline by querying the **Normal Period**.
- Compare error counts and latency distributions.
- **Query Example:**
  ```sql
  SELECT service_name, level, COUNT(*) as error_count 
  FROM normal_logs 
  WHERE level = 'ERROR' 
    AND time >= TIMESTAMP '2025-07-23 14:06:23' 
    AND time < TIMESTAMP '2025-07-23 14:10:23'
  GROUP BY service_name, level 
  ORDER BY error_count DESC 
  LIMIT 20
  ```

Step 5: Deep Dive & Root Cause Identification
- Drill down into the service with the highest error increase or latency spike.
- Trace the error propagation: Is the error internal, or coming from a downstream service?
- Use `trace_id` to correlate logs and traces if needed.
- **Iterate your queries** until you isolate the origin of the fault.

**Final Answer Requirements:**
You MUST provide the final answer in the following exact format:
Root cause service: [service-name]

For example:
Root cause service: ts-food-service
"""

def serialize_message(message):
    """Serialize a LangChain message to a dict."""
    return {
        "type": message.type,
        "content": message.content,
        "additional_kwargs": message.additional_kwargs,
        "response_metadata": message.response_metadata if hasattr(message, "response_metadata") else {},
    }

def main():
    parser = argparse.ArgumentParser(description="RCA Agent")
    parser.add_argument("--query", type=str, help="Initial query to start the agent")
    parser.add_argument("--output", type=str, help="Path to save the output JSON")
    args = parser.parse_args()

    # Check for API Key
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("Please set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.")
        return

    # Create the Deep Agent
    # Explicitly use OpenAI model if ANTHROPIC_API_KEY is not set
    model = None
    if not os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(model="gpt-4o")

    agent = create_deep_agent(
        model=model,
        tools=[list_tables_in_directory, get_schema, query_parquet_files],
        system_prompt=RCA_SYSTEM_PROMPT,
    )

    if args.query:
        print(f"🤖 RCA Agent running with query: {args.query}")
        print("Agent is thinking...")
        
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": args.query}]})
            
            # Print final response
            print(f"\nAgent: {result['messages'][-1].content}")
            
            # Save output if requested
            if args.output:
                messages = result.get("messages", [])
                serialized_history = [serialize_message(msg) for msg in messages]
                
                # Ensure directory exists
                output_dir = os.path.dirname(args.output)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(serialized_history, f, indent=2, ensure_ascii=False)
                print(f"Output saved to {args.output}")
                
        except Exception as e:
            print(f"An error occurred: {e}")
            sys.exit(1)
            
    else:
        # Interactive mode
        print("🤖 RCA Agent Initialized.")
        print("You can now ask the agent to analyze your parquet files.")
        print("Example: 'Analyze the logs in /data/logs to find why the checkout service failed around 10:00 AM.'")
        
        while True:
            try:
                user_input = input("\nUser: ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                
                print("\nAgent is thinking...")
                result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
                print(f"\nAgent: {result['messages'][-1].content}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
