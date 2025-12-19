"""
Quantized Vision Transformer using IRM and DGD methods
Based on Q-ViT: Accurate and Fully Quantized Low-bit Vision Transformer
"""
import torch
import torch.nn as nn
from transformers import ViTForImageClassification
from src.Quant import LinearQ, Conv2dQ
from src.models.irm_layer import IRMLinear

class QuantizedVisionTransformer(nn.Module):
    """
    Quantized Vision Transformer
    Implements IRM (Intra-Rank Modulo) and DGD (Dynamic Gradient Descent) quantization
    """
    
    def __init__(
        self,
        model_name: str = "google/vit-large-patch16-224",
        num_classes: int = 100,
        nbits_w: int = 4,
        nbits_a: int = 4,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.nbits_w = nbits_w
        self.nbits_a = nbits_a
        
        # Load pretrained ViT from HuggingFace
        self.vit = ViTForImageClassification.from_pretrained(model_name)
        
        # Get the base ViT model
        vit_model = self.vit.vit
        
        # Handle Classifier Head
        self.embed_dim = vit_model.config.hidden_size
        
        # If num_classes differs from pretrained, replace the head
        if self.vit.classifier.out_features != num_classes:
            self.vit.classifier = nn.Linear(self.embed_dim, num_classes)
            self.vit.config.num_labels = num_classes
            # Reset label mappings
            self.vit.config.id2label = {i: f"LABEL_{i}" for i in range(num_classes)}
            self.vit.config.label2id = {f"LABEL_{i}": i for i in range(num_classes)}
        
        # Apply Quantization
        self._quantize_model()
        
    def _quantize_model(self):
        """
        Replace layers with quantized versions.
        Uses IRMLinear for Query/Key and LinearQ for others.
        """
        
        # Iterate over all transformer blocks
        for i, block in enumerate(self.vit.vit.encoder.layer):
            
            # --- Quantize Attention ---
            # Query & Key -> Use IRM Layer (Critical for Q-ViT)
            if hasattr(block.attention.attention, 'query'):
                # Query
                orig_layer = block.attention.attention.query
                new_layer = IRMLinear(
                    in_features=orig_layer.in_features,
                    out_features=orig_layer.out_features,
                    bias=orig_layer.bias is not None,
                    nbits_w=self.nbits_w,
                    nbits_a=self.nbits_a
                )
                # Copy weights
                new_layer.linear.weight = orig_layer.weight
                if orig_layer.bias is not None:
                    new_layer.linear.bias = orig_layer.bias
                block.attention.attention.query = new_layer

                # Key
                orig_layer = block.attention.attention.key
                new_layer = IRMLinear(
                    in_features=orig_layer.in_features,
                    out_features=orig_layer.out_features,
                    bias=orig_layer.bias is not None,
                    nbits_w=self.nbits_w,
                    nbits_a=self.nbits_a
                )
                new_layer.linear.weight = orig_layer.weight
                if orig_layer.bias is not None:
                    new_layer.linear.bias = orig_layer.bias
                block.attention.attention.key = new_layer

            # Value & Output -> Use Standard LinearQ
            if hasattr(block.attention.attention, 'value'):
                orig_layer = block.attention.attention.value
                new_layer = LinearQ(
                    in_features=orig_layer.in_features,
                    out_features=orig_layer.out_features,
                    bias=orig_layer.bias is not None,
                    nbits_w=self.nbits_w
                )
                new_layer.weight = orig_layer.weight
                if orig_layer.bias is not None:
                    new_layer.bias = orig_layer.bias
                block.attention.attention.value = new_layer

            if hasattr(block.attention.output, 'dense'):
                orig_layer = block.attention.output.dense
                new_layer = LinearQ(
                    in_features=orig_layer.in_features,
                    out_features=orig_layer.out_features,
                    bias=orig_layer.bias is not None,
                    nbits_w=self.nbits_w
                )
                new_layer.weight = orig_layer.weight
                if orig_layer.bias is not None:
                    new_layer.bias = orig_layer.bias
                block.attention.output.dense = new_layer
            
            # --- Quantize MLP ---
            # Intermediate (FFN 1)
            if hasattr(block.intermediate, 'dense'):
                orig_layer = block.intermediate.dense
                new_layer = LinearQ(
                    in_features=orig_layer.in_features,
                    out_features=orig_layer.out_features,
                    bias=orig_layer.bias is not None,
                    nbits_w=self.nbits_w
                )
                new_layer.weight = orig_layer.weight
                if orig_layer.bias is not None:
                    new_layer.bias = orig_layer.bias
                block.intermediate.dense = new_layer
                
            # Output (FFN 2)
            if hasattr(block.output, 'dense'):
                orig_layer = block.output.dense
                new_layer = LinearQ(
                    in_features=orig_layer.in_features,
                    out_features=orig_layer.out_features,
                    bias=orig_layer.bias is not None,
                    nbits_w=self.nbits_w
                )
                new_layer.weight = orig_layer.weight
                if orig_layer.bias is not None:
                    new_layer.bias = orig_layer.bias
                block.output.dense = new_layer
        
        # --- Quantize Classifier ---
        if isinstance(self.vit.classifier, nn.Linear):
            orig_classifier = self.vit.classifier
            self.vit.classifier = LinearQ(
                in_features=orig_classifier.in_features,
                out_features=orig_classifier.out_features,
                bias=orig_classifier.bias is not None,
                nbits_w=self.nbits_w
            )
            self.vit.classifier.weight = orig_classifier.weight
            if orig_classifier.bias is not None:
                self.vit.classifier.bias = orig_classifier.bias
    
    def forward(self, pixel_values, labels=None):
        """
        Forward pass
        """
        # Get outputs from the wrapped HuggingFace model
        # The internal layers are now quantized
        outputs = self.vit(pixel_values=pixel_values, labels=labels, output_hidden_states=False)
        return outputs
    
    def get_config(self):
        return {
            'num_classes': self.num_classes,
            'nbits_w': self.nbits_w,
            'nbits_a': self.nbits_a,
        }