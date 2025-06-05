from model import main
import logging
import numpy as np
import utils as util
import csv
import os
from collections import defaultdict

def multi_run(name="S1", data_document_path="../KUL_single_single3", num_runs=10, log_path="./result", k=32):
    all_results = []

    log_file = util.makePath(log_path) + "/Multi" + name + ".log"
     

    logger = logging.getLogger("MultiRunLogger")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(fh)

    logger.info(f"Starting multi-run with name: {name}, data path: {data_document_path}, number of runs: {num_runs}")

    for run in range(num_runs):
        print(f"Running iteration {run + 1}/{num_runs}")
        results = main(name=name, data_document_path=data_document_path)
        all_results.append(results)

    # Aggregate accuracies per channel
    accuracy_per_channel = defaultdict(list)

    # Collect accuracy values from each run
    for run_result in all_results:
        for channel, metrics in run_result.items():
            accuracy_per_channel[channel].append(metrics['accuracy'])

    # Compute mean and std dev
    accuracy_stats = {}
    for channel, accuracies in accuracy_per_channel.items():
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        accuracy_stats[channel] = {'mean_accuracy': mean_acc, 'std_accuracy': std_acc}
    
    
    # Print the results
    print("\n=== Accuracy Summary ===")
    for channel in sorted(accuracy_stats.keys(), reverse=True):
        stats = accuracy_stats[channel]
        print(f"{channel} channels: Mean Accuracy = {stats['mean_accuracy']:.4f}, Std Dev = {stats['std_accuracy']:.4f}")

    # Save results to CSV
    # Define output file name
    output_file = 'result.csv'
    file_exists = os.path.exists(output_file)

    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header only if file is new
        if not file_exists:
            header = ['Subject Name']
            for channel in sorted(accuracy_stats.keys(), reverse=True):
                header.append(f'Accuracy@{channel}')
                header.append(f'Std@{channel}')
            writer.writerow(header)

        # Write data row
        row = [name]
        for channel in sorted(accuracy_stats.keys(), reverse=True):
            stats = accuracy_stats[channel]
            row.append(round(stats['mean_accuracy'], 4))
            row.append(round(stats['std_accuracy'], 4))
        writer.writerow(row)

    print(f"✅ Results written to: {output_file}")


if __name__ == "__main__":
    multi_run()

