
from temp import run_subject_analysis  # Update as needed

def run_all_dtu_subjects():
    dataset = "DTU"

    for subject_nr in range(1, 19):
        print(f"\n=== Running analysis for {dataset} subject {subject_nr} ===\n")
        run_subject_analysis(dataset=dataset, subject_nr=subject_nr)

if __name__ == "__main__":
    run_all_dtu_subjects()



