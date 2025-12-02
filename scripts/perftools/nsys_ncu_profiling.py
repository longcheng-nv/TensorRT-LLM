import subprocess
import os
import csv
from typing import List
import csv_parser as CP

def nsys_profile_samplingtopK(
    B_values: List[int],
    N_values: List[int],
    K_values: List[int],
    dtypes: List[str],
    kernel_name: str = "samplingtopK",
    output_dir: str = "./nsys_results",
    csv_summary_name: str = "kernel_summary.csv",
    target_script: str = "./samplingKernelsTest"
) -> None:
    """
    Profile GPU kernels using nsys and save performance results to CSV.
    
    Args:
        B_values: List of batch size values to test
        N_values: List of vocabulary size values to test
        K_values: List of top-K values to test
        dtypes: List of data types to test (e.g., ["float32", "float16"])
        kernel_name: Name of the kernel being profiled
        output_dir: Directory to save profiling results
        csv_summary_name: Name of the output CSV file
        target_script: Path to the executable test script
    
    Returns:
        None. Results are saved to CSV file in output_dir.
    """
    # output directory
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------
    # batch run nsys profile
    # -----------------------------
    print("Start batch collecting GPU kernel performance data...")

    csv_summary_path = os.path.join(output_dir, csv_summary_name)
    with open(csv_summary_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Kernel", "B", "N", "top-K", "dtype", "Average Time(ns)"])

        for B in B_values:
            for N in N_values:
                for K in K_values:
                    for i, dtype in enumerate(dtypes):
                        output_name = f"{kernel_name}_B{B}_N{N}_topK{K}_{dtype}"
                        output_path = os.path.join(output_dir, output_name)

                        # set environment variables
                        env = os.environ.copy()
                        env["TEST_BATCH_SIZE"] = str(B)
                        env["TEST_VOCAB_SIZE"] = str(N)
                        env["TEST_TOP_K"] = str(K)

                        cmd = [
                            "nsys_easy",
                            "-o",
                            f"{output_path}",
                            target_script,
                            f"--gtest_filter=TopKSamplingKernelTest/{i}.parameterizedTopK",
                            "--force-export=true"
                        ]
                        cmd_stats = [
                            "nsys",
                            "stats",
                            "--report",
                            "cuda_gpu_trace",
                            f"{output_path}.nsys-rep",
                            "--format",
                            "csv",
                            "--output", f"{output_path}.csv",
                            "--force-export=true"
                        ]
                        # nsys stats --report cuda_gpu_trace *.nsys-rep --format csv --output *.csv

                        print(f"Run command: {' '.join(cmd)}")
                        print(f"Environment variables: TEST_BATCH_SIZE={B}, TEST_VOCAB_SIZE={N}, TEST_TOP_K={K}")
                        try:
                            result = subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)
                            print(f"Result: {result.stdout}")
                            #
                            result_stats = subprocess.run(cmd_stats, check=True, capture_output=True, text=True)
                            print(f"Stats result: {result_stats.stdout}")
                            #
                            kernel_name = "topKStage1"
                            Avg = CP.find_kernel_metric(file_path=f"{output_path}.csv_cuda_gpu_trace.csv", kernel_name=kernel_name, metric_name="Duration")
                            writer.writerow([kernel_name, B, N, K, dtype, Avg])
                            #
                            kernel_name = "topKStage2Sampling"
                            Avg = CP.find_kernel_metric(file_path=f"{output_path}.csv_cuda_gpu_trace.csv", kernel_name=kernel_name, metric_name="Duration")
                            writer.writerow([kernel_name, B, N, K, dtype, Avg])
                        except subprocess.CalledProcessError as e:
                            print(f"Task failed: {output_name}, error: {e}")

    print("All nsys profile finished.")
    print(f"Summary data saved to {csv_summary_path}")
    return csv_summary_path

def ncu_profile_samplingtopK(
    B_values: List[int],
    N_values: List[int],
    K_values: List[int],
    dtypes: List[str],
    kernel_name: str = "samplingtopK",
    output_dir: str = "./ncu_results",
    target_script: str = "./samplingKernelsTest"
) -> None:
    """
    Profile GPU kernels using ncu and save performance results to CSV.
    
    Args:
        B_values: List of batch size values to test
        N_values: List of vocabulary size values to test
        K_values: List of top-K values to test
        dtypes: List of data types to test (e.g., ["float32", "float16"])
        kernel_name: Name of the kernel being profiled
        output_dir: Directory to save profiling results
        csv_summary_name: Name of the output CSV file
        target_script: Path to the executable test script
    
    Returns:
        None. Results are saved to CSV file in output_dir.
    """
    # output directory
    os.makedirs(output_dir, exist_ok=True)
    # -----------------------------
    # batch run ncu profile
    # -----------------------------
    print("Start batch collecting GPU kernel performance data...")
    for B in B_values:
            for N in N_values:
                for K in K_values:
                    for i, dtype in enumerate(dtypes):
                        output_name = f"{kernel_name}_B{B}_N{N}_topK{K}_{dtype}"
                        output_path = os.path.join(output_dir, output_name)

                        # set environment variables
                        env = os.environ.copy()
                        env["TEST_BATCH_SIZE"] = str(B)
                        env["TEST_VOCAB_SIZE"] = str(N)
                        env["TEST_TOP_K"] = str(K)

                        cmd = [
                            "ncu",
                            "-o",
                            f"{output_path}",
                            target_script,
                            f"--gtest_filter=TopKSamplingKernelTest/{i}.parameterizedTopK"
                        ]
                        print(f"Run command: {' '.join(cmd)}")
                        print(f"Environment variables: TEST_BATCH_SIZE={B}, TEST_VOCAB_SIZE={N}, TEST_TOP_K={K}")
                        try:
                            result = subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)
                        except subprocess.CalledProcessError as e:
                            print(f"Task failed: {output_name}, error: {e}")

    print("All ncu profile finished.")

if __name__ == "__main__":
    # -----------------------------
    # parameter configurations: Deepseek V3/R1
    # -----------------------------
    kernel_name = "samplingtopK"
    output_dir = "./nsys_results"
    output_dir_ncu = "./ncu_results"
    csv_summary_name = "kernel_summary.csv"
    target_script = "./samplingKernelsTest"
    #
    B_values = [1, 2, 4, 8, 16, 32, 64]
    N_values = [129280]
    K_values = [50]
    dtypes = ["float32", "float16"]
    #
    # Run nsys profiling with default parameters
    nsys_profile_samplingtopK(
        B_values=B_values,
        N_values=N_values,
        K_values=K_values,
        dtypes=dtypes,
        kernel_name=kernel_name,
        output_dir=output_dir,
        csv_summary_name=csv_summary_name,
        target_script=target_script
    )
    #
    # Run ncu profiling with default parameters
    ncu_profile_samplingtopK(
        B_values=B_values,  
        N_values=N_values,
        K_values=K_values,
        dtypes=dtypes,
        kernel_name=kernel_name,
        output_dir=output_dir_ncu,
        target_script=target_script
    )
