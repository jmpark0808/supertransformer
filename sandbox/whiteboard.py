import torch.nn as nn
import torch

class TokenDropout(nn.Module):
    def __init__(self, p: float = 0.1):
        """
        Token-level dropout: randomly zero out entire token features with probability p.
        
        Args:
        - p (float): Probability of dropping out each token.
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        """
        Args:
        - x: Input tensor of shape (B, N, D)
        
        Returns:
        - Tensor with some tokens zeroed out.
        """
        if not self.training or self.p == 0.0:
            return x  # No dropout during evaluation mode

        B, N, D = x.shape
        # Generate random mask of shape (B, N, 1)
        mask = (torch.rand(B, N, 1, device=x.device) > self.p).float()
        return x * mask  # Zero out selected tokens
    
from PIL import Image
from torchvision.transforms import ToTensor
image1 = Image.open('/home/eddie/Datasets/DUTS/DUTS-TE/Image/ILSVRC2012_test_00000003.jpg').resize((224, 224))
image2 = Image.open('/home/eddie/Datasets/DUTS/DUTS-TE/Image/ILSVRC2012_test_00000023.jpg').resize((224, 224))
image3 = Image.open('/home/eddie/Datasets/DUTS/DUTS-TE/Image/ILSVRC2012_test_00000025.jpg').resize((224, 224))
image4 = Image.open('/home/eddie/Datasets/DUTS/DUTS-TE/Image/ILSVRC2012_test_00000034.jpg').resize((224, 224))
tt = ToTensor()

image_tensor1 = tt(image1)
image_tensor2 = tt(image2)
image_tensor3 = tt(image3)
image_tensor4 = tt(image4)


batch = torch.stack((image_tensor1, image_tensor2, image_tensor3, image_tensor4), dim=0).reshape(4, 3, -1).permute(0, 2, 1)

dropout = TokenDropout(0.5)
batch_dropout = dropout(batch).reshape(4, 224, 224, 3).permute(0, 3, 1, 2)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 2)
ax[0,0].imshow(batch_dropout[0].permute(1, 2, 0).detach().numpy())
ax[0,1].imshow(batch_dropout[1].permute(1, 2, 0).detach().numpy())
ax[1,0].imshow(batch_dropout[2].permute(1, 2, 0).detach().numpy())
ax[1,1].imshow(batch_dropout[3].permute(1, 2, 0).detach().numpy())
plt.show()

