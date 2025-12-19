"""
PyTorch Lightning module for Quantized ViT-L with DGD
FIXED: Added Debugging for Zero DGD Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy, F1Score
from transformers import ViTForImageClassification
from src.models.quantized_vit import QuantizedVisionTransformer
import numpy as np

class LitQuantizedViT(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "google/vit-large-patch16-224",
        num_classes: int = 100,
        nbits_w: int = 4,
        nbits_a: int = 4,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.0,
        warmup_epochs: int = 10,
        max_epochs: int = 30,
        batch_size: int = 32,
        optimizer: str = "adamw",
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. Student
        self.model = QuantizedVisionTransformer(
            model_name=model_name,
            num_classes=num_classes,
            nbits_w=nbits_w,
            nbits_a=nbits_a,
        )
        
        # 2. Teacher
        print("Loading Teacher model for DGD...")
        self.teacher = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        self.student_features = {}
        self.teacher_features = {}
        self.hooks_registered = False
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average='weighted')
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.optimizer_name = optimizer

    def setup_hooks(self):
        """Robust Hook Registration"""
        if self.hooks_registered: return

        # Closure to capture output
        def get_activation(name, storage):
            def hook(model, input, output):
                # Handle HF output tuples
                if isinstance(output, tuple): 
                    output = output[0]
                # Clone to ensure we keep the data
                storage[name] = output.clone()
            return hook

        print(f"DEBUG: Registering DGD hooks...")
        
        # Student Hooks (Quantized Layers)
        s_count = 0
        for i, layer in enumerate(self.model.vit.vit.encoder.layer):
            # Hook the IRM Linear layers in Attention
            layer.attention.attention.query.register_forward_hook(
                get_activation(f'q_{i}', self.student_features))
            layer.attention.attention.key.register_forward_hook(
                get_activation(f'k_{i}', self.student_features))
            s_count += 1

        # Teacher Hooks (Standard Layers)
        t_count = 0
        for i, layer in enumerate(self.teacher.vit.encoder.layer):
            layer.attention.attention.query.register_forward_hook(
                get_activation(f'q_{i}', self.teacher_features))
            layer.attention.attention.key.register_forward_hook(
                get_activation(f'k_{i}', self.teacher_features))
            t_count += 1
            
        self.hooks_registered = True
        print(f"DEBUG: Hooks registered on {s_count} Student layers and {t_count} Teacher layers.")

    def compute_dgd_loss(self):
        """Computes Similarity Matrix Loss"""
        
        # --- DIAGNOSTIC PRINT (Only runs for first 5 steps) ---
        if self.global_step < 5:
            print(f"\n[Step {self.global_step}] DGD Debug:")
            print(f"  Student Keys captured: {len(self.student_features)}")
            print(f"  Teacher Keys captured: {len(self.teacher_features)}")
            if len(self.student_features) > 0:
                print(f"  Sample Key: {list(self.student_features.keys())[0]}")
                print(f"  Shape: {self.student_features[list(self.student_features.keys())[0]].shape}")
        # -----------------------------------------------------

        if len(self.student_features) == 0:
            return torch.tensor(0.0, device=self.device)

        dgd_loss = 0.0
        pairs_computed = 0
        
        for key in self.student_features:
            if key not in self.teacher_features: continue
                
            s_feat = self.student_features[key]
            t_feat = self.teacher_features[key]
            
            # Similarity Matrix (Gram Matrix)
            # [Batch, N, D] x [Batch, D, N] -> [Batch, N, N]
            s_gram = torch.bmm(s_feat, s_feat.transpose(1, 2))
            t_gram = torch.bmm(t_feat, t_feat.transpose(1, 2))
            
            # Normalize (Important for DGD stability)
            s_gram = F.normalize(s_gram, p=2, dim=(1, 2))
            t_gram = F.normalize(t_gram, p=2, dim=(1, 2))
            
            dgd_loss += F.mse_loss(s_gram, t_gram)
            pairs_computed += 1
            
        # Scaling the loss to be visible (MSE is often very small)
        # We multiply by 1000 to make it comparable to CE loss
        dgd_loss = dgd_loss * 1000.0
        
        return dgd_loss

    def on_train_start(self):
        self.setup_hooks()
        self.teacher = self.teacher.to(self.device)

    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values, labels)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        
        # 1. Clear previous features
        self.student_features.clear()
        self.teacher_features.clear()
        
        # 2. Teacher Forward (Populates teacher_features)
        with torch.no_grad():
            self.teacher(images)
            
        # 3. Student Forward (Populates student_features)
        outputs = self.model(pixel_values=images, labels=labels)
        logits = outputs.logits
        
        # 4. Losses
        ce_loss = self.criterion(logits, labels)
        dgd_loss = self.compute_dgd_loss()
        
        total_loss = ce_loss + dgd_loss
        
        # 5. Logging
        acc = self.train_acc(logits, labels)
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_dgd_loss', dgd_loss, prog_bar=True) 
        self.log('train_ce_loss', ce_loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(pixel_values=images, labels=labels)
        logits = outputs.logits
        loss = self.criterion(logits, labels)
        acc = self.val_acc(logits, labels)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(pixel_values=images, labels=labels)
        logits = outputs.logits
        loss = self.criterion(logits, labels)
        acc = self.test_acc(logits, labels)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay
            )
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            else:
                progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}