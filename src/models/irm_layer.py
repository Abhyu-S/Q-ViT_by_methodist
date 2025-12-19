import torch
import torch.nn as nn
from src.Quant import LinearQ, ActQ

class IRMLinear(nn.Module):
    """
    Implements Information Rectification Module (IRM) for Query/Key layers in Q-ViT.
    
    Based on Q-ViT Paper Eq (8):
    Reshapes the distribution of queries/keys using learnable parameters (gamma, beta)
    to maximize information entropy before quantization.
    """
    def __init__(self, in_features, out_features, bias=True, nbits_w=4, nbits_a=4):
        super().__init__()
        
        # 1. Base Linear Layer with Weight Quantization
        # We use LinearQ for the weight quantization capabilities
        self.linear = LinearQ(
            in_features=in_features, 
            out_features=out_features, 
            bias=bias, 
            nbits_w=nbits_w
        )
        
        # 2. IRM Learnable Parameters
        # Gamma and Beta reshape the distribution (Mean/Var) of the activation
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))
        self.epsilon = 1e-5
        
        # 3. Activation Quantizer
        # Quantizes the rectified activation
        self.act_quant = ActQ(in_features=out_features, nbits_a=nbits_a)

    def forward(self, x):
        """
        Forward pass with Information Rectification
        """
        # Step 1: Linear Projection (Weight Quantized internally in LinearQ)
        # x shape: [Batch, SeqLen, OutFeatures]
        x = self.linear(x)
        
        # Step 2: Information Rectification (IRM)
        # Normalize the distribution to standard normal, then rescale with learnable params
        # This allows the model to "rectify" the information flow before damaging quantization
        
        # Calculate statistics over the last dimension (embedding dim)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.epsilon)
        
        # Apply Rectification: (x - mu) / std * gamma + beta
        # Note: Paper Eq(8) uses division by gamma, but standard implementation 
        # (like LayerNorm/BN) uses multiplication. Mathematically equivalent for learning.
        x_rect = ((x - mean) / std) * self.gamma + self.beta
        
        # Step 3: Quantize the rectified activations
        x_out = self.act_quant(x_rect)
        
        return x_out