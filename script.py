import csv
from temp import run_subject_analysis  # Update as needed

def run_all_dtu_subjects():
    dataset = "DTU"
    all_results = []

    for subject_nr in range(1, 19):
        print(f"\n=== Running analysis for {dataset} subject {subject_nr} ===\n")
        try:
            result = run_subject_analysis(dataset=dataset, subject_nr=subject_nr)
            all_results.append(result)
        except Exception as e:
            print(f"⚠️ Failed to process subject {subject_nr}: {e}")
            all_results.append({
                "subject": subject_nr,
                "dataset": dataset,
                "mean_acc": "ERR",
                "std_acc": "ERR",
                "mean_shap_acc": "ERR",
                "std_shap_acc": "ERR"
            })

    # Save results to CSV
    output_file = "dtu_all_subjects_results.csv"
    with open(output_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n✅ Results saved to: {output_file}")

if __name__ == "__main__":
    run_all_dtu_subjects()



