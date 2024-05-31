import numpy as np
import torch

class iouEval:
    """
    Class to evaluate IoU and related metrics for image segmentation tasks.

    Attributes:
        nClasses (int): Number of classes in the segmentation task.
    """
    def __init__(self, nClasses):
        """
        Initializes the IOUEval class with the number of classes and resets metrics.

        Parameters:
            nClasses (int): Number of segmentation classes.
        """
        self.nClasses = nClasses
        self.reset()

    def reset(self):
        """
        Resets evaluation metrics to start a new evaluation.
        """
        self.overall_acc = 0
        self.per_class_acc = np.zeros(self.nClasses, dtype=np.float32)
        self.per_class_iu = np.zeros(self.nClasses, dtype=np.float32)
        self.mean_IOU = 0
        self.batch_count = 0

    def fast_hist(self, label_true, label_pred):
        """
        Creates a histogram of the ground truth vs predicted labels for IoU calculation.

        Parameters:
            label_true (array): Ground truth labels.
            label_pred (array): Predicted labels.

        Returns:
            ndarray: Histogram matrix of size (nClasses, nClasses).
        """
        mask = (label_true >= 0) & (label_true < self.nClasses)
        hist = np.bincount(
            self.nClasses * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.nClasses**2
        ).reshape(self.nClasses, self.nClasses)
        return hist

    def add_batch(self, predictions, ground_truths):
        """
        Process a batch of predictions and ground truths to update the metrics.

        Parameters:
            predictions (Tensor): Predicted labels.
            ground_truths (Tensor): Actual labels.
        """
        predictions = predictions.cpu().numpy().flatten()
        ground_truths = ground_truths.cpu().numpy().flatten()

        hist = self.compute_hist(predictions, ground_truths)

        self.overall_acc += np.diag(hist).sum() / (hist.sum() + np.finfo(float).eps)
        self.per_class_acc += np.diag(hist) / (hist.sum(1) + np.finfo(float).eps)
        self.per_class_iu += np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + np.finfo(float).eps)
        self.mean_IOU += np.nanmean(self.per_class_iu)
        self.batch_count += 1

    def compute_hist(self, predictions, ground_truths):
        """
        Compute the histogram for a single batch of predictions and ground truths.

        Parameters:
            predictions (array): Flattened array of predictions.
            ground_truths (array): Flattened array of ground truths.

        Returns:
            ndarray: Computed histogram for the batch.
        """
        return self.fast_hist(ground_truths, predictions)

    def get_metric(self):
        """
        Computes the final averaged metrics over all batches.

        Returns:
            tuple: Tuple containing overall accuracy, per-class accuracy,
                   per-class intersection over union, and mean intersection over union.
        """
        overall_acc = self.overall_acc / self.batch_count
        per_class_acc = self.per_class_acc / self.batch_count
        per_class_iu = self.per_class_iu / self.batch_count
        mean_IOU = self.mean_IOU / self.batch_count

        return overall_acc, per_class_acc, per_class_iu, mean_IOU