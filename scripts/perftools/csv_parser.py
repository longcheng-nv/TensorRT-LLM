#!/usr/bin/env python3
"""
CSV/Excel data processing utility for kernel performance analysis.
"""

import pandas as pd
from typing import Optional, Union, List


def find_kernel_metric(
    file_path: str,
    kernel_name: str,
    metric_name: str,
    header_row: int = 0
) -> Optional[Union[float, str]]:
    """
    Find a specific metric value for a given kernel in a CSV/Excel file.
    
    Args:
        file_path: Path to the CSV or Excel file
        kernel_name: Kernel name string to search for (partial match)
        metric_name: Column name (metric) to look up (case-insensitive)
        header_row: Row number containing column headers (default: 0)
    
    Returns:
        The value at the intersection of the kernel row and metric column,
        or None if not found.
    
    Example:
        >>> value = find_kernel_metric("data.csv", "topKStage1", "Duration")
        >>> print(value)
        43392.0
    """
    try:
        # Step 1: Read the CSV/Excel file
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path, header=header_row)
        else:
            df = pd.read_csv(file_path, header=header_row)
        
        print(f"Loaded file: {file_path}")
        print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
        
        # Step 2: Find row containing the kernel name
        row_id = None
        for idx, row in df.iterrows():
            # Check if kernel_name exists in any cell of this row
            row_str = ' '.join(str(cell) for cell in row.values)
            if kernel_name in row_str:
                row_id = idx
                print(f"Found kernel '{kernel_name}' at row {row_id}")
                break
        
        if row_id is None:
            print(f"Warning: Kernel '{kernel_name}' not found in file")
            return None
        
        # Step 3: Find column containing the metric name (case-insensitive)
        col_id = None
        metric_name_lower = metric_name.lower()
        for col_idx, col_name in enumerate(df.columns):
            if metric_name_lower in str(col_name).lower():
                col_id = col_name  # Use column name for pandas access
                print(f"Found metric '{metric_name}' at column '{col_name}' (index {col_idx})")
                break
        
        if col_id is None:
            print(f"Warning: Metric '{metric_name}' not found in columns")
            return None
        
        # Step 4: Get the value at (row_id, col_id)
        value = df.loc[row_id, col_id]
        print(f"Value at row {row_id}, column '{col_id}': {value}")
        
        return value
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None
    except Exception as e:
        print(f"Error processing file: {e}")
        return None


def find_all_kernel_metrics(
    file_path: str,
    kernel_name: str,
    metric_names: List[str],
    header_row: int = 0
) -> dict:
    """
    Find multiple metric values for a given kernel.
    
    Args:
        file_path: Path to the CSV or Excel file
        kernel_name: Kernel name string to search for
        metric_names: List of column names (metrics) to look up
        header_row: Row number containing column headers
    
    Returns:
        Dictionary mapping metric names to their values.
    """
    results = {}
    
    try:
        # Read file once
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path, header=header_row)
        else:
            df = pd.read_csv(file_path, header=header_row)
        
        # Find kernel row
        row_id = None
        for idx, row in df.iterrows():
            row_str = ' '.join(str(cell) for cell in row.values)
            if kernel_name in row_str:
                row_id = idx
                break
        
        if row_id is None:
            print(f"Warning: Kernel '{kernel_name}' not found")
            return results
        
        # Find each metric
        for metric_name in metric_names:
            metric_lower = metric_name.lower()
            for col_name in df.columns:
                if metric_lower in str(col_name).lower():
                    results[metric_name] = df.loc[row_id, col_name]
                    break
        
        return results
        
    except Exception as e:
        print(f"Error: {e}")
        return results


# Example usage
if __name__ == "__main__":
    # Example 1: Find single metric
    print("=" * 60)
    print("Example 1: Find Duration for topKStage1")
    print("=" * 60)
    
    value = find_kernel_metric(
        file_path="./nsys_results/samplingtopK_B1_N129280_topK50_float32.csv_cuda_gpu_trace.csv",
        kernel_name="topKStage1",
        metric_name="Duration"
    )
    print(f"\nResult: {value}")
    
    # Example 2: Find multiple metrics
    print("\n" + "=" * 60)
    print("Example 2: Find multiple metrics for topKStage1")
    print("=" * 60)
    
    metrics = find_all_kernel_metrics(
        file_path="./nsys_results/samplingtopK_B1_N129280_topK50_float32.csv_cuda_gpu_trace.csv",
        kernel_name="topKStage1",
        metric_names=["Duration", "Start"]
    )
    print(f"\nResults: {metrics}")

