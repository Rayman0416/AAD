from model import main
import logging
import numpy as np
import utils as util

def multi_run(name="S1", data_document_path="../KUL_single_single3", num_runs=10, log_path="./result", k=32):
    all_losses = []
    all_accuracies = []
    all_loss_reduced = []
    all_accuracy_reduced = []

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
        loss, accuracy, loss_reduced, accuracy_reduced = main(name=name, data_document_path=data_document_path)

        all_losses.append(loss)
        all_accuracies.append(accuracy)
        all_loss_reduced.append(loss_reduced)
        all_accuracy_reduced.append(accuracy_reduced)

        logger.info(f"Run {run + 1}: Loss = {loss}, Accuracy = {accuracy}, Reduced Loss = {loss_reduced}, Reduced Accuracy = {accuracy_reduced}")
    
    mean_loss = np.mean(all_losses)
    mean_accuracy = np.mean(all_accuracies)
    mean_loss_reduced = np.mean(all_loss_reduced)
    mean_accuracy_reduced = np.mean(all_accuracy_reduced)

    logger.info(f"Mean Loss: {mean_loss:.4f}, Mean Accuracy: {mean_accuracy:.4f}, Mean Reduced Loss: {mean_loss_reduced:.4f}, Mean Reduced Accuracy: {mean_accuracy_reduced:.4f}")


if __name__ == "__main__":
    multi_run()

