import sqlite3
import pandas as pd
import os

# --- Adjust Pandas Display Options ---
# Set the display width to a large number (or None to try and auto-detect)
# Adjust this number based on your actual terminal width if needed
pd.set_option('display.width', 1000)
# Ensure all columns are displayed
pd.set_option('display.max_columns', None)
# Optionally, increase max column width if individual cells are truncated
# pd.set_option('display.max_colwidth', None)
# ------------------------------------

def query_transcript_db(db_path, query):
    """
    Connects to the SQLite DB, executes a query, and returns the results
    as a pandas DataFrame.
    """
    conn = None # Initialize connection to None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        # Get column names from the description of the cursor that executed the query
        columns = [description[0] for description in cursor.description] if cursor.description else []
        df = pd.DataFrame(results, columns=columns)
        return df
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        # Return an empty DataFrame in case of error
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()
    finally:
        # Ensure the connection is closed even if errors occur
        if conn:
            conn.close()

if __name__ == "__main__":
    # Default database path in the data directory
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    db_path = os.path.join(data_dir, "AndrejKarpathy.db")
    # query first 10 rows
    query = "SELECT * FROM transcripts LIMIT 10"

    # Function now returns the DataFrame directly
    df = query_transcript_db(db_path, query)

    # Print the resulting DataFrame (Pandas will now use the updated display settings)
    if not df.empty:
        print(df)
    else:
        print("Query returned no results or an error occurred.")
