import torch.nn
import torch
from .misc import gaussian_kernel
from .metrics import ssim


class SSIM_loss(torch.nn.Module):
    def __init__(
        self,
        spatial_dimensions=3,
        channels=1,
        size=[9, 9, 9],
        sigma=[0.5, 0.5, 0.5],
        dynamic_range=1,
        k=[0.01, 0.03],
        reduction="mean",
        mask=False,
        device=torch.device("cuda"),
    ) -> None:
        """SSIM loss module

        Parameters
        ----------
        spatial_dimensions : int, optional
            target spatial dimensions, by default 3
        channels: int, optional
            target channel dimensions, by default 1
        size : int, optional
            kernel size, by default 5
        sigma : int, optional
            kernel sigma, by default 3
        dynamic_range : int, optional
            intensity dynamic range of tensor and prediction tensors, by default 1
        k : list, optional
            Stabilizer constants for SSIM, by default [0.01, 0.03]
        reduction : str or None, optional
            reduction mode used, "sum" or "mean" (or None for no reductin), by default "mean"
        mask: bool, optional
            if weights are provided during the forward loop and mask==True the values where weight is zero (in all channels) are excluded from the ssim_val before reducing, by default False
        device: torch.device
            device on which the kernel is loaded
        """
        super().__init__()
        # Check spatial_dimensions parameter
        if spatial_dimensions not in [1, 2, 3]:
            raise ValueError("spatial dimensions must be either 1, 2, or 3.")
        # Check channels parameter
        if not isinstance(channels, int) or channels <= 0:
            raise TypeError("channels must be a postive integer")
        # Check reduction parameter
        if reduction not in [None, "mean", "sum"]:
            raise ValueError('reduction must be either "mean", "sum" or None')
        # Chek mask parameter
        if not isinstance(mask, bool):
            raise TypeError("mask must be a boolean")

        # Generate kernel
        self.kernel = gaussian_kernel(size, sigma)
        self.kernel /= torch.sum(self.kernel)
        self.kernel = self.kernel.unsqueeze(0).unsqueeze(0)
        kernel_repeat_shape = [1] * (2 + spatial_dimensions)
        kernel_repeat_shape[0] = channels
        self.kernel = self.kernel.repeat(kernel_repeat_shape).to(device)

        # Save module variables
        self.dynamic_range = dynamic_range
        self.k = k
        self.mask = mask
        self.reduction = reduction

    def forward(self, target, prediction, weights=None):
        """Returns SSIM loss between target and prediction

        Parameters
        ----------
        target : torch.Tensor
            target tensor of shape (B, C, *) where * is one, two, or three spatial dimensions
        prediction : torch.Tensor
            prediction tensor of shape target.shape 
        weights : torch.Tensor, optional
            weights tensor of shape target.shape used to weight the loss, by default None
        
        Returns
        -------
        torch.Tensor
            SSIM loss between target and prediction
        """
        # Check target parameter
        if not isinstance(target, torch.Tensor):
            raise TypeError("target must be a tensor")
        elif len(target.shape) > 5:
            raise ValueError("target must have a max. of 3 spatial dimensions")
        elif torch.min(target) < 0:
            raise ValueError("target values must be non-negative")
        # Check prediction parameter
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("prediction must be a tensor")
        elif prediction.shape != target.shape:
            raise ValueError("prediction must same shape as target")
        elif torch.min(prediction) < 0:
            raise ValueError("prediction values must be non-negative")
        elif type(prediction) != type(target):
            raise TypeError("prediction must be the same type as target")
        # Check weights parameter
        if weights is not None:
            if type(weights) != type(target):
                raise TypeError("weights must be the same type as target")
            elif weights.shape != target.shape:
                raise ValueError("weights must have the same shape as target")

        # Obtain non-reduced ssim_index
        ssim_index = ssim(
            target=target,
            prediction=prediction,
            kernel=self.kernel,
            dynamic_range=self.dynamic_range,
            k=self.k,
            reduction=None,
        )[0]
        # Apply weights if required
        if weights is not None:
            # Centre-crop weights to match ssim_index
            c = (torch.tensor(target.shape) - torch.tensor(ssim_index.shape)) / 2
            c_low = torch.floor(c).int()
            c_high = torch.ceil(c).int()
            weights = weights[
                :,
                :,
                c_low[2] : -c_high[2],
                c_low[3] : -c_high[3],
                c_low[4] : -c_high[4],
            ]
            # Apply weights
            ssim_index *= weights
        # Average ssim_index over channels
        ssim_index = torch.mean(ssim_index, dim=1).flatten(1)
        # Mask results if required
        if self.mask and weights is not None:
            # Generate masks
            mask = (weights.sum(dim=1) > 0).int().flatten(1)
            # Apply masks
            ssim_index = torch.sum(ssim_index * mask, dim=1)
            # Compensate for mask sizes (with 1e-10 to avoid division by zero)
            ssim_index /= torch.sum(mask, dim=1) + 1e-10
        # If masking is not required, average ssim_index for each batch element
        else:
            # Average ssim_index
            ssim_index = torch.mean(ssim_index, dim=1)
        # Reduce ssim_index
        if self.reduction == "mean":
            ssim_index = ssim_index.mean()
        elif self.reduction == "sum":
            ssim_index = ssim_index.sum()
        # Return ssim_loss
        return 1 - ssim_index


class KLD_loss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        """Kullback-Leibler divergence loss module

        Parameters
        ----------
        reduction : str or None, optional
            reduction mode used, "sum" or "mean" (or None for no reductin), by default "mean"
        """
        super(KLD_loss, self).__init__()
        # Check reduction parameter
        if reduction not in [None, "mean", "sum"]:
            raise ValueError('reduction must be either "mean", "sum", or None')
        self.reduction = reduction

    def forward(self, mu, log_var):
        """Returns the Kullback-Leibler divergence loss from mu and log_var

        Parameters
        ----------
        mu : torch.Tensor
            mean output of encoder
        log_var : torch.Tensor
            logarithmic variance of output of encoder

        Returns
        -------
        torch.Tensor
            Kullback-Leibler divergence loss
        """
        # Calculate loss
        loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1)
        # Reduce loss
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        return loss


class MSE_loss(torch.nn.Module):
    def __init__(self, reduction="mean", mask=False) -> None:
        """MSE loss module

        Parameters
        ----------
        reduction : str or None, optional
            reduction mode used, "sum" or "mean" (or None for no reductin), by default "mean"
        mask: bool, optional
            if weights are provided during the forward loop and mask==True the values where weight is zero (in all channels) are excluded from the loss before reducing, by default False
        """
        super().__init__()
        # Check reduction parameter
        if reduction not in [None, "mean", "sum"]:
            raise ValueError('reduction must be either "mean" or "sum"')
        # Chek mask parameter
        if not isinstance(mask, bool):
            raise TypeError("mask must be a boolean")

        self.reduction = reduction
        self.mask = mask

    def forward(self, target, prediction, weights=None):
        """Returns MSE loss between target and prediction

        Parameters
        ----------
        target : torch.Tensor
            target tensor of shape (B, C, *) where * is one, two, or three spatial dimensions
        prediction : torch.Tensor
            prediction tensor of shape target.shape 
        weights : torch.Tensor, optional
            weights tensor of shape target.shape used to weight the loss, by default None

        Returns
        -------
        torch.Tensor
            MSE loss between target and prediction

        """
        # Check target parameter
        if not isinstance(target, torch.Tensor):
            raise TypeError("target must be a tensor")
        elif len(target.shape) > 5:
            raise ValueError("target must have a max. of 3 spatial dimensions")
        # Check prediction parameter
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("prediction must be a tensor")
        elif prediction.shape != target.shape:
            raise ValueError("prediction must same shape as target")
        elif type(prediction) != type(target):
            raise TypeError("prediction must be the same type as target")
        # Check weights parameter
        if weights is not None:
            if type(weights) != type(target):
                raise TypeError("weights must be the same type as target")
            elif weights.shape != target.shape:
                raise ValueError("weights must have the same shape as target")

        # Calculate squared error
        sq_error = torch.square(prediction - target)

        # Apply weights if required
        if weights is not None:
            # Apply weights
            sq_error *= weights

        # Average sq_error over channels
        sq_error = torch.mean(sq_error, dim=1).flatten(1)

        # Mask results if required
        if self.mask and weights is not None:
            # Generate masks
            mask = (weights.sum(dim=1) > 0).int().flatten(1)
            # Apply masks and average results
            loss = torch.sum(sq_error * mask, dim=1) / torch.sum(mask, dim=1)

        # If masking is not required, average loss for each batch element
        else:
            # Average loss
            loss = torch.mean(sq_error, dim=1)

        # Reduce loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        # Return loss
        return loss


class MAE_loss(torch.nn.Module):
    def __init__(self, reduction="mean", mask=False) -> None:
        """MAE loss module

        Parameters
        ----------
        reduction : str or None, optional
            reduction mode used, "sum" or "mean" (or None for no reductin), by default "mean"
        mask: bool, optional
            if weights are provided during the forward loop and mask==True the values where weight is zero (in all channels) are excluded from the loss before reducing, by default False
        """
        super().__init__()
        # Check reduction parameter
        if reduction not in [None, "mean", "sum"]:
            raise ValueError('reduction must be either "mean" or "sum"')
        # Chek mask parameter
        if not isinstance(mask, bool):
            raise TypeError("mask must be a boolean")

        self.reduction = reduction
        self.mask = mask

    def forward(self, target, prediction, weights=None):
        """Returns MAE loss between target and prediction

        Parameters
        ----------
        target : torch.Tensor
            target tensor of shape (B, C, *) where * is one, two, or three spatial dimensions
        prediction : torch.Tensor
            prediction tensor of shape target.shape 
        weights : torch.Tensor, optional
            weights tensor of shape target.shape used to weight the loss, by default None

        Returns
        -------
        torch.Tensor
            MAE loss between target and prediction

        """
        # Check target parameter
        if not isinstance(target, torch.Tensor):
            raise TypeError("target must be a tensor")
        elif len(target.shape) > 5:
            raise ValueError("target must have a max. of 3 spatial dimensions")
        # Check prediction parameter
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("prediction must be a tensor")
        elif prediction.shape != target.shape:
            raise ValueError("prediction must same shape as target")
        elif type(prediction) != type(target):
            raise TypeError("prediction must be the same type as target")
        # Check weights parameter
        if weights is not None:
            if type(weights) != type(target):
                raise TypeError("weights must be the same type as target")
            elif weights.shape != target.shape:
                raise ValueError("weights must have the same shape as target")

        # Calculate absolute error
        abs_err = torch.abs(prediction - target)
        # Apply weights if required
        if weights is not None:
            # Apply weights
            abs_err *= weights

        # Average abs_error over channels
        abs_err = torch.mean(abs_err, dim=1).flatten(1)

        # Mask results if required
        if self.mask and weights is not None:
            # Generate masks
            mask = (weights.sum(dim=1) > 0).int().flatten(1)

            # Apply masks and average results
            loss = torch.sum(abs_err * mask, dim=1) / torch.sum(mask, dim=1)

        # If masking is not required, average loss for each batch element
        else:
            # Average loss
            loss = torch.mean(abs_err, dim=1)
        # Reduce loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        # Return loss
        return loss


class BCE_loss(torch.nn.Module):
    def __init__(self, reduction="mean", mask=False) -> None:
        """BCE loss module

        Parameters
        ----------
        reduction : str or None, optional
            reduction mode used, "sum" or "mean" (or None for no reductin), by default "mean"
        mask: bool, optional
            if weights are provided during the forward loop and mask==True the values where weight is zero (in all channels) are excluded from the loss before reducing, by default False
        """
        super().__init__()
        # Check reduction parameter
        if reduction not in [None, "mean", "sum"]:
            raise ValueError('reduction must be either "mean" or "sum"')
        # Chek mask parameter
        if not isinstance(mask, bool):
            raise TypeError("mask must be a boolean")

        self.reduction = reduction
        self.module = torch.nn.BCELoss(reduction="none")
        self.mask = mask

    def forward(self, target, prediction, weights=None):
        """Returns BCE loss between target and prediction

            Parameters
            ----------
            target : torch.Tensor
                target tensor of shape (B, C, *) where * is one, two, or three spatial dimensions
            prediction : torch.Tensor
                prediction tensor of shape target.shape 
            weights : torch.Tensor, optional
                weights tensor of shape target.shape used to weight the loss, by default None

            Returns
            -------
            torch.Tensor
                BCE loss between target and prediction

            """
        # Check target parameter
        if not isinstance(target, torch.Tensor):
            raise TypeError("target must be a tensor")
        elif len(target.shape) > 5:
            raise ValueError("target must have a max. of 3 spatial dimensions")
        elif torch.min(target) < 0 or torch.max(target) > 1:
            raise ValueError("target values must lie between 0 and 1")
        # Check prediction parameter
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("prediction must be a tensor")
        elif prediction.shape != target.shape:
            raise ValueError("prediction must same shape as target")
        elif torch.min(prediction) < 0 or torch.max(prediction) > 1:
            raise ValueError("prediction values must lie between 0 and 1")
        elif type(prediction) != type(target):
            raise TypeError("prediction must be the same type as target")
        # Check weights parameter
        if weights is not None:
            if type(weights) != type(target):
                raise TypeError("weights must be the same type as target")
            elif weights.shape != target.shape:
                raise ValueError("weights must have the same shape as target")

        # Calculate binary cross-entropy
        bce = self.module(prediction, target)  # The order must be (pred, target)

        # Apply weights if required
        if weights is not None:
            # Apply weights
            bce *= weights

        # Average abs_error over channels
        bce = torch.mean(bce, dim=1).flatten(1)

        # Mask results if required
        if self.mask and weights is not None:
            # Generate masks
            mask = (weights.sum(dim=1) > 0).int().flatten(1)

            # Apply masks and average results
            loss = torch.sum(bce * mask, dim=1) / torch.sum(mask, dim=1)

        # If masking is not required, average loss for each batch element
        else:
            # Average loss
            loss = torch.mean(bce, dim=1)
        # Reduce loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        # Return loss
        return loss


class Dice_loss(torch.nn.Module):
    def __init__(self, reduction="mean") -> None:
        """Dice loss module

        Parameters
        ----------
        reduction : str or None, optional
            reduction mode used, "sum" or "mean" (or None for no reductin), by default "mean"
        """
        super().__init__()
        # Check reduction parameter
        if reduction not in [None, "mean", "sum"]:
            raise ValueError('reduction must be either "mean" or "sum"')
        self.reduction = reduction

    def forward(self, target, prediction, smooth=1):
        """ Returns the Sørensen–Dice coefficient loss between target and prediction

        Args:
            target (torch.tensor): Target to calcualte DSC against
            prediction (torch.tensor): Prediction being assessed
            smooth (int, optional): Smoothing factor of DSC. Defaults to 1.

        Returns:
            torch.Tensor: Dice loss
        """

        # Flatten tensors
        target = target.view(target.shape[0], -1)
        prediction = prediction.view(prediction.shape[0], -1)

        # Calculate intersection
        intersection = (target * prediction).sum()

        # Calculate loss
        loss = 1 - (2.0 * intersection + smooth) / (
            target.sum() + prediction.sum() + smooth
        )
        # Reduce loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        # Return loss
        return loss
