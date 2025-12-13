import math
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for 5D regression.
    
    A fully-connected neural network with configurable architecture
    for learning mappings f(x1, ..., x5) = y.
    
    Args:
        in_dim: Number of input features (default: 5).
        hidden: Number of neurons per hidden layer (default: 64).
        depth: Number of hidden layers (default: 3).
        out_dim: Number of output dimensions (default: 1).
        activation: Activation function class (default: nn.SiLU).
    
    Example:
        >>> model = MLP(hidden=64, depth=3)
        >>> x = torch.randn(32, 5)
        >>> y = model(x)
        >>> y.shape
        torch.Size([32, 1])
    """
    
    def __init__(
        self,
        in_dim: int = 5,
        hidden: int = 64,
        depth: int = 3,
        out_dim: int = 1,
        activation: type = nn.SiLU
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.depth = depth
        self.out_dim = out_dim
        
        # Build network layers
        layers = [nn.Linear(in_dim, hidden), activation()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), activation()]
        layers += [nn.Linear(hidden, out_dim)]
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.net(x)
    
    def count_parameters(self) -> int:
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FiveDRegressor:
    """
    A scikit-learn style wrapper for the MLP model.
    
    Provides fit(X, y) and predict(X) interface for easy integration
    with existing workflows.
    
    Args:
        hidden_layers: List of neurons per layer, e.g., [64, 32, 16].
        learning_rate: Learning rate for optimizer (default: 5e-3).
        max_epochs: Maximum training epochs (default: 200).
        batch_size: Training batch size (default: 256).
        patience: Early stopping patience (default: 20).
        weight_decay: L2 regularization strength (default: 1e-6).
        val_fraction: Fraction of data for validation (default: 0.15).
        random_state: Random seed for reproducibility (default: 42).
        verbose: Whether to print training progress (default: True).
    
    Example:
        >>> model = FiveDRegressor(hidden_layers=[64, 32, 16])
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(
        self,
        hidden_layers: Optional[List[int]] = None,
        learning_rate: float = 5e-3,
        max_epochs: int = 200,
        batch_size: int = 256,
        patience: int = 20,
        weight_decay: float = 1e-6,
        val_fraction: float = 0.15,
        random_state: int = 42,
        verbose: bool = True
    ):
        self.hidden_layers = hidden_layers or [64, 32, 16]
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.weight_decay = weight_decay
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.verbose = verbose
        
        # Will be set during training
        self.model: Optional[nn.Module] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_val_loss: float = float("inf")
        
        # Normalization statistics
        self.x_mean: Optional[np.ndarray] = None
        self.x_std: Optional[np.ndarray] = None
        self.y_mean: Optional[float] = None
        self.y_std: Optional[float] = None
    
    def _build_model(self) -> nn.Module:
        """Build the MLP model based on hidden_layers configuration."""
        layers = []
        in_features = 5
        
        for hidden in self.hidden_layers:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.SiLU())
            in_features = hidden
        
        layers.append(nn.Linear(in_features, 1))
        
        return nn.Sequential(*layers)
    
    def _normalize_x(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize input features."""
        if fit:
            self.x_mean = X.mean(axis=0)
            self.x_std = X.std(axis=0) + 1e-8  # Avoid division by zero
        return (X - self.x_mean) / self.x_std
    
    def _normalize_y(self, y: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize target values."""
        if fit:
            self.y_mean = y.mean()
            self.y_std = y.std() + 1e-8
        return (y - self.y_mean) / self.y_std
    
    def _denormalize_y(self, y: np.ndarray) -> np.ndarray:
        """Denormalize target values back to original scale."""
        return y * self.y_std + self.y_mean
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "FiveDRegressor":
        """
        Train the model on the provided data.
        
        Args:
            X: Feature array of shape (n_samples, 5).
            y: Target array of shape (n_samples,).
            
        Returns:
            Self for method chaining.
        """
        # Set random seed for reproducibility
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Ensure correct dtypes
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        # Normalize data
        X_norm = self._normalize_x(X, fit=True)
        y_norm = self._normalize_y(y, fit=True)
        
        # Convert to tensors
        X_tensor = torch.from_numpy(X_norm)
        y_tensor = torch.from_numpy(y_norm).unsqueeze(1)
        
        # Create dataset and split
        dataset = TensorDataset(X_tensor, y_tensor)
        n_total = len(dataset)
        n_val = int(self.val_fraction * n_total)
        n_val = max(1, min(n_val, n_total - 1)) 
        n_train = n_total - n_val
        
        train_ds, val_ds = random_split(
            dataset, 
            [n_train, n_val],
            generator=torch.Generator().manual_seed(self.random_state)
        )
        
        train_loader = DataLoader(
            train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            drop_last=False
        )
        val_loader = DataLoader(
            val_ds, 
            batch_size=1024, 
            shuffle=False
        )
        
        # Build and move model to device
        self.model = self._build_model().to(self.device)
        
        # Setup optimizer, scheduler, and loss
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.max_epochs
        )
        loss_fn = nn.MSELoss()
        
        # Training loop
        best_state = None
        no_improve = 0
        self.train_losses = []
        self.val_losses = []
        
        for epoch in range(1, self.max_epochs + 1):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * xb.size(0)
            
            scheduler.step()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred = self.model(xb)
                    val_loss += loss_fn(pred, yb).item() * xb.size(0)
            
            train_loss /= len(train_ds)
            val_loss /= len(val_ds)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Early stopping check
            if val_loss < self.best_val_loss - 1e-6:
                self.best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            
            # Logging
            if self.verbose and (epoch % 20 == 0 or no_improve == 1):
                lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch:4d} | train MSE {train_loss:.5f} | "
                      f"val MSE {val_loss:.5f} | lr {lr:.3e}")
            
            # Early stopping
            if no_improve >= self.patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch}. "
                          f"Best val MSE: {self.best_val_loss:.5f}")
                break
        
        # Load best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        self.model = self.model.cpu()
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature array of shape (n_samples, 5).
            
        Returns:
            Predictions array of shape (n_samples,).
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")
        
        # Ensure correct dtype
        X = np.asarray(X, dtype=np.float32)
        
        # Normalize input
        X_norm = self._normalize_x(X, fit=False)
        X_tensor = torch.from_numpy(X_norm)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            pred_norm = self.model(X_tensor).numpy().flatten()
        
        # Denormalize output
        return self._denormalize_y(pred_norm)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Compute evaluation metrics on the given data.
        
        Args:
            X: Feature array of shape (n_samples, 5).
            y: Target array of shape (n_samples,).
            
        Returns:
            Dictionary with 'mse', 'rmse', 'mae', and 'r2' scores.
        """
        y_pred = self.predict(X)
        y = np.asarray(y)
        
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - y_pred))
        
        # R² score
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2)
        }
    
    def get_params(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return {
            "hidden_layers": self.hidden_layers,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "weight_decay": self.weight_decay,
            "val_fraction": self.val_fraction,
            "random_state": self.random_state
        }
    
    def save(self, filepath: str) -> None:
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path to save the model.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")
        
        state = {
            "model_state_dict": self.model.state_dict(),
            "params": self.get_params(),
            "x_mean": self.x_mean,
            "x_std": self.x_std,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
            "hidden_layers": self.hidden_layers
        }
        torch.save(state, filepath)
    
    def load(self, filepath: str) -> "FiveDRegressor":
        """
        Load a trained model from a file.
        
        Args:
            filepath: Path to the saved model.
            
        Returns:
            Self for method chaining.
        """
        state = torch.load(filepath, map_location="cpu", weights_only=False)
        
        self.hidden_layers = state["hidden_layers"]
        self.x_mean = state["x_mean"]
        self.x_std = state["x_std"]
        self.y_mean = state["y_mean"]
        self.y_std = state["y_std"]
        
        self.model = self._build_model()
        self.model.load_state_dict(state["model_state_dict"])
        
        return self