import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss2d(nn.Module):
    """
    This class defines a cross-entropy loss for 2D images.
    """
    def __init__(self, weight=None):
        """
        Initializes the CrossEntropyLoss2d.
        
        Parameters:
            weight (Tensor, optional): A manual rescaling weight given to each class.
                                       If given, has to be a Tensor of size `C`
        """
        super(CrossEntropyLoss2d, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss2d(weight=weight)

    def forward(self, outputs, targets):
        """
        Forward pass of the loss function.

        Parameters:
            outputs (Tensor): The class logits from the model. Shape (N, C, H, W)
                              where N is the batch size, C is the number of classes,
                              H and W are the height and width of the input tensors.
            targets (Tensor): The ground truth labels. Shape (N, H, W)

        Returns:
            Tensor: The computed loss value.
        """
        # Applying log_softmax on the class dimension (dim=1)
        outputs = F.log_softmax(outputs, dim=1)
        return self.loss(outputs, targets)
