import torch
import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: int=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class LinearHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class VectorPerceiver(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_heads=4, dropout=0.1):
        """
        Args:
            dim: The dimensionality of your input vectors.
            output_dim: The dimensionality of the output vector.
            num_heads: Number of attention heads (must divide dim).
            dropout: Dropout rate for regularization.
        """
        super().__init__()
        
        # 1. The Latent Query: This is the "target" vector that will
        # compress the 3 inputs into 1.
        self.latent_query = nn.Parameter(torch.randn(1, 1, input_dim))
        
        # 2. Cross-Attention: Latent queries the input vectors
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=input_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        # 3. Lightweight Feed-Forward Network (FFN) to refine the result
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(input_dim)
        self.ln3 = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Dropout(dropout)
        )
        self.linear_projector = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 3, dim)
        Returns:
            Compressed vector of shape (batch_size, dim)
        """
        b, n, d = x.shape
        
        # Expand latent query to match batch size
        # shape: (batch_size, 1, dim)
        latent = self.latent_query.expand(b, -1, -1)
        
        # Step 1: Cross-Attention
        # Query = latent, Key/Value = input vectors (x)
        attn_out, _ = self.cross_attn(
            query=self.ln1(latent), 
            key=x, 
            value=x
        )
        latent = latent + attn_out # Residual connection
        
        # Step 2: Feed-Forward
        ffn_out = self.ffn(self.ln2(latent))
        latent = latent + ffn_out # Residual connection
        
        # Return the single vector (remove the sequence dimension)
        return self.linear_projector(self.ln3(latent.squeeze(1)))

# # --- Usage Example ---
# if __name__ == "__main__":
#     # Parameters
#     batch_size = 8
#     vector_dim = 128
    
#     # Create the model
#     model = VectorPerceiver(dim=vector_dim)
    
#     # Mock data: 3 vectors per batch item
#     input_vectors = torch.randn(batch_size, 3, vector_dim)
    
#     # Compress
#     compressed_vector = model(input_vectors)
    
#     print(f"Input shape: {input_vectors.shape}")      # [8, 3, 128]
#     print(f"Output shape: {compressed_vector.shape}") # [8, 128]