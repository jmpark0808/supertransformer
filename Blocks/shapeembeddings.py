import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchedShapeEmbeddingCNN(nn.Module):
    def __init__(self, embedding_dim, num_superpixels):
        super().__init__()
        self.K = num_superpixels
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, embedding_dim)
        )

    def forward(self, seg_maps):  # (B, H, W) with labels 0..K-1 per image
        B, H, W = seg_maps.shape
        K = self.K

        # One-hot encode to (B, K, H, W)
        seg_maps_onehot = F.one_hot(seg_maps, num_classes=K).permute(0, 3, 1, 2).float()  # (B, K, H, W)

        # Flatten B and K to process with CNN: (B*K, 1, H, W)
        masks = seg_maps_onehot.view(B * K, 1, H, W)

        # Encode masks
        embeddings = self.encoder(masks)  # (B*K, D)

        # Reshape back to (B, K, D)
        D = embeddings.shape[-1]
        embeddings = embeddings.view(B, K, D)

        return embeddings
    

if __name__ == "__main__":
    # Instantiate model
    model = BatchedShapeEmbeddingCNN(embedding_dim=64, num_superpixels=3136)

    # Dummy input: B=4, K=20, H=W=224 → total B*K masks
    B, K, H, W = 1, 3136, 224, 224
    dummy_input = torch.randint(0, 3136, (B, H, W)).long()

    # FLOPs analysis
    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(model, dummy_input)
    print("FLOPs:", flops.total(), "FLOPs")
    
