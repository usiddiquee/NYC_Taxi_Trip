import time
import psutil
import pandas as pd
import polars as pl
import duckdb
import dask.dataframe as dd
# import pyspark.pandas as ps  # Koalas (pandas API on Spark)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

app = FastAPI()

# Helper function to measure execution time and CPU usage
def measure_performance(func, *args, **kwargs):
    print(f"[INFO] Running: {func.__name__}")
    
    start_time = time.time()
    process = psutil.Process()
    cpu_usage_before = process.cpu_percent(interval=None)

    try:
        result = func(*args, **kwargs)
    except Exception as e:
        print(f"[ERROR] {func.__name__} failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    cpu_usage_after = process.cpu_percent(interval=None)
    end_time = time.time()

    execution_time = round(end_time - start_time, 4)
    cpu_usage = round(cpu_usage_after - cpu_usage_before, 2)

    print(f"[INFO] {func.__name__} completed - Time: {execution_time}s, CPU: {cpu_usage}%")

    return result, {
        "execution_time": execution_time,
        "cpu_usage": cpu_usage
    }

# File reading methods
def read_pandas(file_path):
    return pd.read_parquet(file_path)

# def read_koalas(file_path):
#     return ps.read_parquet(file_path)

def read_polars(file_path):
    return pl.read_parquet(file_path)

def read_duckdb(file_path):
    return duckdb.read_parquet(file_path)

def read_dask(file_path):
    return dd.read_parquet(file_path).compute()

# Pure Python (Loops)
def read_pure_python(file_path):
    import pyarrow.parquet as pq
    table = pq.read_table(file_path)
    data = table.to_pydict()
    return data

# Python Multiprocessing
def read_multiprocessing(file_path):
    import pyarrow.parquet as pq
    table = pq.read_table(file_path)
    data = table.to_pydict()

    def process_chunk(chunk):
        return {k: v for k, v in chunk.items()}

    with Pool(cpu_count()) as pool:
        chunks = [dict(zip(data.keys(), [data[k][i:i + 1000] for k in data])) for i in range(0, len(data[list(data.keys())[0]]), 1000)]
        results = pool.map(process_chunk, chunks)

    return results

# Python Vectorized Operations
def read_vectorized(file_path):
    import pyarrow.parquet as pq
    table = pq.read_table(file_path)
    return table.to_pandas()

# Mapping for library selection
LIBRARY_FUNCTIONS = {
    "pandas": read_pandas,
    # "koalas": read_koalas,
    "polars": read_polars,
    "duckdb": read_duckdb,
    "dask": read_dask,
    # "multiprocessing": read_multiprocessing,
    "vectorized": read_vectorized
}

# Request body model for preprocessing
class PreprocessRequest(BaseModel):
    file_path: str
    library: str
    drop_na: Optional[bool] = False
    select_columns: Optional[List[str]] = None
    filter_column: Optional[str] = None
    filter_value: Optional[str] = None

@app.get("/benchmark")
def benchmark_file(file_path: str):
    results = {}
    
    for library, func in LIBRARY_FUNCTIONS.items():
        try:
            _, stats = measure_performance(func, file_path)
            results[library] = stats
        except Exception as e:
            results[library] = {"error": str(e)}
    
    return {"benchmark_results": results}

@app.post("/preprocess")
def preprocess_file(request: PreprocessRequest):
    file_path = request.file_path
    library = request.library.lower()

    print(f"\n[INFO] Received preprocessing request for: {file_path} using {library}")

    if library not in LIBRARY_FUNCTIONS:
        print(f"[ERROR] Unsupported library: {library}")
        raise HTTPException(status_code=400, detail=f"Library '{library}' not supported.")

    try:
        print(f"[INFO] Loading file using {library}...")
        df, load_stats = measure_performance(LIBRARY_FUNCTIONS[library], file_path)

        if df is None:
            raise HTTPException(status_code=500, detail="Failed to load data.")

        print("[INFO] Checking column names...")
        if isinstance(df, (pd.DataFrame, dd.DataFrame)):  # Pandas, Dask, or Koalas
            df.columns = df.columns.str.strip()  # Strip spaces from column names
        elif isinstance(df, pl.DataFrame):  # Polars
            df = df.rename({col: col.strip() for col in df.columns})
        elif isinstance(df, duckdb.DuckDBPyRelation):  # DuckDB
            df = duckdb.sql("SELECT * FROM df")  # Ensure it's a valid queryable table
        elif isinstance(df, (dict, list)):  # Pure Python or Multiprocessing
            df = pd.DataFrame(df)  # Convert to Pandas DataFrame for consistency

        print("Columns available:", df.columns)

        # Check if filter column exists
        if request.filter_column and request.filter_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{request.filter_column}' not found in dataset.")

        # Preprocessing logic
        print("[INFO] Preprocessing started...")

        def preprocess():
            df_processed = df

            # Apply filtering first
            if request.filter_column and request.filter_value:
                print(f"[INFO] Filtering {request.filter_column} where value = {request.filter_value}")

                if isinstance(df_processed, (pd.DataFrame, dd.DataFrame)):
                    df_processed[request.filter_column] = df_processed[request.filter_column].astype(str)
                    df_processed = df_processed[df_processed[request.filter_column] == request.filter_value]
                
                elif isinstance(df_processed, pl.DataFrame):  # Polars
                    df_processed = df_processed.with_columns(df_processed[request.filter_column].cast(pl.Utf8))
                    df_processed = df_processed.filter(df_processed[request.filter_column] == request.filter_value)

                elif isinstance(df_processed, duckdb.DuckDBPyRelation):  # DuckDB
                    df_processed = duckdb.sql(f"SELECT * FROM df_processed WHERE {request.filter_column} = '{request.filter_value}'")

            # Drop NaN values
            if request.drop_na:
                print("[INFO] Dropping NA values...")
                if isinstance(df_processed, (pd.DataFrame, dd.DataFrame)):
                    df_processed.dropna(inplace=True)
                elif isinstance(df_processed, pl.DataFrame):
                    df_processed = df_processed.drop_nulls()

            # Select specific columns
            if request.select_columns:
                print(f"[INFO] Selecting columns: {request.select_columns}")
                missing_cols = [col for col in request.select_columns if col not in df_processed.columns]
                if missing_cols:
                    raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")
                
                if isinstance(df_processed, (pd.DataFrame, dd.DataFrame, pl.DataFrame)):
                    df_processed = df_processed[request.select_columns]
                elif isinstance(df_processed, duckdb.DuckDBPyRelation):
                    df_processed = duckdb.sql(f"SELECT {', '.join(request.select_columns)} FROM df_processed")

            return df_processed

        df_processed, process_stats = measure_performance(preprocess)

        print("[INFO] Preprocessing completed successfully.")

        return {
            "message": "Preprocessing completed",
            "processed_rows": len(df_processed),
            "load_time": load_stats["execution_time"],
            "load_cpu_usage": load_stats["cpu_usage"],
            "process_time": process_stats["execution_time"],
            "process_cpu_usage": process_stats["cpu_usage"]
        }

    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/benchmark_preprocess")
def benchmark_preprocess(request: PreprocessRequest):
    results = {}
    
    for library in LIBRARY_FUNCTIONS.keys():
        try:
            request.library = library
            response = preprocess_file(request)
            results[library] = {
                "load_time": response["load_time"],
                "load_cpu_usage": response["load_cpu_usage"],
                "process_time": response["process_time"],
                "process_cpu_usage": response["process_cpu_usage"]
            }
        except Exception as e:
            results[library] = {"error": str(e)}

    # Plotting the results
    libraries = list(results.keys())
    load_times = [results[lib].get("load_time", 0) for lib in libraries]
    process_times = [results[lib].get("process_time", 0) for lib in libraries]
    load_cpu_usages = [results[lib].get("load_cpu_usage", 0) for lib in libraries]
    process_cpu_usages = [results[lib].get("process_cpu_usage", 0) for lib in libraries]

    # Plot Load Time
    plt.figure(figsize=(10, 5))
    plt.bar(libraries, load_times, color='skyblue')
    plt.ylabel('Time (s)')
    plt.title('Load Time by Library')
    load_time_chart = "load_time_chart.png"
    plt.savefig(load_time_chart)
    plt.close()

    # Plot Load CPU Usage
    plt.figure(figsize=(10, 5))
    plt.bar(libraries, load_cpu_usages, color='lightgreen')
    plt.ylabel('CPU Usage (%)')
    plt.title('Load CPU Usage by Library')
    load_cpu_chart = "load_cpu_usage_chart.png"
    plt.savefig(load_cpu_chart)
    plt.close()

    # Plot Process Time
    plt.figure(figsize=(10, 5))
    plt.bar(libraries, process_times, color='lightcoral')
    plt.ylabel('Time (s)')
    plt.title('Process Time by Library')
    process_time_chart = "process_time_chart.png"
    plt.savefig(process_time_chart)
    plt.close()

    # Plot Process CPU Usage
    plt.figure(figsize=(10, 5))
    plt.bar(libraries, process_cpu_usages, color='gold')
    plt.ylabel('CPU Usage (%)')
    plt.title('Process CPU Usage by Library')
    process_cpu_chart = "process_cpu_usage_chart.png"
    plt.savefig(process_cpu_chart)
    plt.close()

    print(f"[INFO] Charts saved as {load_time_chart}, {load_cpu_chart}, {process_time_chart}, {process_cpu_chart}")

    return {
        "benchmark_preprocess_results": results,
        "charts_saved_as": {
            "load_time": load_time_chart,
            "load_cpu_usage": load_cpu_chart,
            "process_time": process_time_chart,
            "process_cpu_usage": process_cpu_chart
        }
    }