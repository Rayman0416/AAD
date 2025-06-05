from model import main
import logging
import numpy as np
import utils as util

def multi_run(name="S1", data_document_path="../KUL_single_single3", num_runs=10, log_path="./result", k=32):
    all_losses = []
    all_accuracies = []
    all_loss_reduced_32 = []
    all_accuracy_reduced_32 = []
    all_loss_reduced_16 = []
    all_accuracy_reduced_16 = []

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
        loss, accuracy, loss_reduced_32, accuracy_reduced_32, loss_reduced_16, accuracy_reduced_16 = main(name=name, data_document_path=data_document_path)

        all_losses.append(loss)
        all_accuracies.append(accuracy)
        all_loss_reduced_32.append(loss_reduced_32)
        all_accuracy_reduced_32.append(accuracy_reduced_32)
        all_loss_reduced_16.append(loss_reduced_16)
        all_accuracy_reduced_16.append(accuracy_reduced_16)

        logger.info(f"Run {run + 1}: Loss = {loss}, Accuracy = {accuracy}, Loss Reduced 32 = {loss_reduced_32}, Accuracy Reduced 32 = {accuracy_reduced_32}, Loss Reduced 16 = {loss_reduced_16}, Accuracy Reduced 16 = {accuracy_reduced_16}")
    
    mean_loss = np.mean(all_losses)
    mean_accuracy = np.mean(all_accuracies)
    mean_loss_reduced_32 = np.mean(all_loss_reduced_32)
    mean_accuracy_reduced_32 = np.mean(all_accuracy_reduced_32)
    mean_loss_reduced_16 = np.mean(all_loss_reduced_16)
    mean_accuracy_reduced_16 = np.mean(all_accuracy_reduced_16)

    logger.info(f"Mean Loss: {mean_loss:.4f}, Mean Accuracy: {mean_accuracy:.4f}, Mean Reduced Loss: {mean_loss_reduced_32:.4f}, Mean Reduced Accuracy: {mean_accuracy_reduced_32:.4f}, Mean Reduced Loss 16: {mean_loss_reduced_16:.4f}, Mean Reduced Accuracy 16: {mean_accuracy_reduced_16:.4f}")


if __name__ == "__main__":
    multi_run()

