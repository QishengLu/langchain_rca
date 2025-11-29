import json
import os
from datetime import datetime
from pathlib import Path
from typing import Union, List, Any

import duckdb

TOKEN_LIMIT = 5000

def _serialize_datetime(obj: Any) -> Any:
    """Convert datetime objects to ISO format strings for JSON serialization"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: _serialize_datetime(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_datetime(item) for item in obj]
    else:
        return obj

def _estimate_token_count(text: str) -> int:
    """Estimate token count using character-based approximation."""
    average_chars_per_token = 3
    return (len(text) + average_chars_per_token - 1) // average_chars_per_token

def _enforce_token_limit(payload: str, context: str) -> str:
    """Ensure payload stays within the token budget before returning"""
    token_estimate = _estimate_token_count(payload)
    if token_estimate <= TOKEN_LIMIT:
        return payload

    try:
        current_size = len(json.loads(payload)) if payload.startswith("[") else None
    except json.JSONDecodeError:
        current_size = None
        
    suggested_limit = None
    if current_size:
        ratio = TOKEN_LIMIT / token_estimate
        suggested_limit = max(1, int(current_size * ratio * 0.8))

    suggestion_parts = [
        "The query result is too large. Please adjust your query:",
        "  • Reduce the LIMIT value" + (f" (try LIMIT {suggested_limit})" if suggested_limit else ""),
        "  • Filter rows with WHERE clauses to reduce result size",
        "  • Select only necessary columns instead of SELECT *",
        "  • Use aggregation (COUNT, SUM, AVG) instead of retrieving raw rows",
    ]

    warning = {
        "error": "Result exceeds token budget",
        "context": context,
        "estimated_tokens": token_estimate,
        "token_limit": TOKEN_LIMIT,
        "rows_returned": current_size,
        "suggested_limit": suggested_limit,
        "suggestion": "\n".join(suggestion_parts),
    }
    return json.dumps(warning, ensure_ascii=False, indent=2)

def _validate_parquet_files(parquet_files: Union[str, List[str]]) -> List[str]:
    """Validate parquet files exist and return as list."""
    if isinstance(parquet_files, str):
        parquet_files = [parquet_files]

    for file_path in parquet_files:
        if not os.path.exists(file_path):
             if not Path(file_path).exists():
                raise FileNotFoundError(
                    f"Parquet file not found: {file_path}\n"
                    f"Please check the file path and ensure the file exists. "
                    f"You may use 'list_tables_in_directory' to discover available parquet files."
                )
    return parquet_files

def list_tables_in_directory(directory: str) -> str:
    """
    List all parquet files in the given directory.
    
    Args:
        directory: The directory path to search for .parquet files.
        
    Returns:
        A JSON string containing a list of parquet filenames found.
    """
    try:
        if not os.path.exists(directory):
            return f"Error: Directory '{directory}' does not exist."
        
        files = [f for f in os.listdir(directory) if f.endswith('.parquet')]
        return json.dumps(files, indent=2)
    except Exception as e:
        return f"Error listing files: {str(e)}"

def get_schema(parquet_files: Union[str, List[str]]) -> str:
    """
    Get the schema (column names and types) of parquet file(s).
    
    Args:
        parquet_files: Path(s) to parquet file(s).
        
    Returns:
        A JSON string describing the columns and their data types for each file.
    """
    try:
        parquet_files = _validate_parquet_files(parquet_files)
        conn = duckdb.connect(":memory:")
        schemas = {}
        
        for file_path in parquet_files:
            file_name = Path(file_path).name
            try:
                query = f"DESCRIBE SELECT * FROM read_parquet('{file_path}')"
                result = conn.execute(query).fetchall()
                # Format: column_name, column_type, ...
                columns = [{"column_name": r[0], "column_type": r[1]} for r in result]
                schemas[file_name] = columns
            except Exception as e:
                schemas[file_name] = f"Error reading schema: {str(e)}"
                
        return json.dumps(schemas, indent=2)
    except Exception as e:
        return f"Error getting schema: {str(e)}"

def query_parquet_files(parquet_files: Union[str, List[str]], query: str, limit: int = 50) -> str:
    """
    Query parquet files using SQL syntax for data analysis and exploration.
    
    The tool automatically registers the provided parquet files as tables (views) 
    using their filenames (without extension) as table names.
    
    Args:
        parquet_files: Path(s) to parquet file(s) to be queried.
        query: SQL query to execute. Use table names corresponding to filenames.
        limit: Maximum number of records to return (default 50).
        
    Returns:
        JSON string of query results.
    """
    try:
        parquet_files = _validate_parquet_files(parquet_files)
        conn = duckdb.connect(":memory:")
        table_names = set()
        
        # Register views
        for file_path in parquet_files:
            base_name = Path(file_path).stem
            table_name = base_name
            counter = 1
            while table_name in table_names:
                table_name = f"{base_name}_{counter}"
                counter += 1
            table_names.add(table_name)
            # Use read_parquet for safety and flexibility
            conn.execute(f"CREATE VIEW {table_name} AS SELECT * FROM read_parquet('{file_path}')")
            
        # Execute query
        result = conn.execute(query).fetchall()
        if not result:
             return "Query executed successfully but returned no results."

        columns = [desc[0] for desc in conn.description]
        rows = [dict(zip(columns, row)) for row in result]
        serialized_rows = _serialize_datetime(rows)
        
        if len(serialized_rows) > limit:
            serialized_rows = serialized_rows[:limit]
            
        result_json = json.dumps(serialized_rows, ensure_ascii=False, indent=2)
        return _enforce_token_limit(result_json, "query_parquet_files")
        
    except Exception as e:
        # Improved error handling from attachment
        error_msg = str(e)
        if "syntax error" in error_msg.lower() or "parser error" in error_msg.lower():
            return (f"SQL syntax error: {error_msg}\n"
                    f"Query: {query}\n"
                    f"Available tables: {list(table_names)}")
        elif "catalog" in error_msg.lower() or "table" in error_msg.lower():
             return (f"Table reference error: {error_msg}\n"
                     f"Available tables: {list(table_names)}")
        else:
            return f"Query execution failed: {error_msg}"
