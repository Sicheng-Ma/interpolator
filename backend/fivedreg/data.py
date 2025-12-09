import pickle
from pathlib import Path
from typing import Tuple, Optional, Union

import numpy as np
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler

class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class FiveDDataset:
    """
    A class to handle 5D datasets for regression tasks.
    
    This class provides methods for loading, validating, preprocessing,
    and splitting datasets with 5 input features.
    
    Attributes:
        X (np.ndarray): Feature array of shape (n_samples, 5).
        y (np.ndarray): Target array of shape (n_samples,).
        scaler (StandardScaler): Fitted scaler for feature standardization.
        is_standardized (bool): Whether features have been standardized.
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize the dataset with features and targets.
        
        Args:
            X: Feature array of shape (n_samples, 5).
            y: Target array of shape (n_samples,).
            
        Raises:
            DataValidationError: If data validation fails.
        """
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.scaler: Optional[StandardScaler] = None
        self.is_standardized = False
        
        self._validate()
    
    def _validate(self) -> None:
        """
        Validate the dataset dimensions and contents.
        
        Raises:
            DataValidationError: If validation fails.
        """
        # Check X dimensions
        if self.X.ndim != 2:
            raise DataValidationError(
                f"X must be 2-dimensional, got {self.X.ndim} dimensions"
            )
        
        if self.X.shape[1] != 5:
            raise DataValidationError(
                f"X must have 5 features, got {self.X.shape[1]} features"
            )
        
        # Check y dimensions
        if self.y.ndim != 1:
            # Try to flatten if it's a column vector
            if self.y.ndim == 2 and self.y.shape[1] == 1:
                self.y = self.y.flatten()
            else:
                raise DataValidationError(
                    f"y must be 1-dimensional, got {self.y.ndim} dimensions"
                )
        
        # Check matching samples
        if self.X.shape[0] != self.y.shape[0]:
            raise DataValidationError(
                f"X and y must have the same number of samples. "
                f"X has {self.X.shape[0]}, y has {self.y.shape[0]}"
            )
        
        # Check for empty dataset
        if self.X.shape[0] == 0:
            raise DataValidationError("Dataset cannot be empty")
    
    def handle_missing_values(
        self, 
        strategy: str = "mean",
        fill_value: Optional[float] = None
    ) -> "FiveDDataset":
        """
        Handle missing values (NaN) in the dataset.
        
        Args:
            strategy: Strategy for handling missing values.
                - "mean": Replace with column mean
                - "median": Replace with column median
                - "drop": Remove rows with missing values
                - "fill": Replace with a constant value
            fill_value: Value to use when strategy is "fill".
            
        Returns:
            Self for method chaining.
            
        Raises:
            ValueError: If invalid strategy or missing fill_value.
        """
        # Check for NaN in X
        x_nan_mask = np.isnan(self.X)
        y_nan_mask = np.isnan(self.y)
        
        has_x_nan = np.any(x_nan_mask)
        has_y_nan = np.any(y_nan_mask)
        
        if not has_x_nan and not has_y_nan:
            return self  # No missing values
        
        if strategy == "drop":
            # Remove rows with any NaN
            row_mask = ~(np.any(x_nan_mask, axis=1) | y_nan_mask)
            self.X = self.X[row_mask]
            self.y = self.y[row_mask]
            
        elif strategy == "mean":
            if has_x_nan:
                col_means = np.nanmean(self.X, axis=0)
                for col in range(self.X.shape[1]):
                    self.X[x_nan_mask[:, col], col] = col_means[col]
            if has_y_nan:
                self.y[y_nan_mask] = np.nanmean(self.y)
                
        elif strategy == "median":
            if has_x_nan:
                col_medians = np.nanmedian(self.X, axis=0)
                for col in range(self.X.shape[1]):
                    self.X[x_nan_mask[:, col], col] = col_medians[col]
            if has_y_nan:
                self.y[y_nan_mask] = np.nanmedian(self.y)
                
        elif strategy == "fill":
            if fill_value is None:
                raise ValueError("fill_value must be provided when strategy is 'fill'")
            self.X[x_nan_mask] = fill_value
            self.y[y_nan_mask] = fill_value
            
        else:
            raise ValueError(
                f"Invalid strategy '{strategy}'. "
                f"Must be one of: 'mean', 'median', 'drop', 'fill'"
            )
        
        # Re-validate after handling missing values
        self._validate()
        return self
    
    def standardize(self, scaler: Optional[StandardScaler] = None) -> "FiveDDataset":
        """
        Standardize features to zero mean and unit variance.
        
        Args:
            scaler: Optional pre-fitted scaler. If None, a new scaler
                   will be fitted on the data.
                   
        Returns:
            Self for method chaining.
        """
        if scaler is not None:
            self.scaler = scaler
            self.X = self.scaler.transform(self.X)
        else:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)
        
        self.is_standardized = True
        return self
    
    def split(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: Optional[int] = 42
    ) -> Tuple["FiveDDataset", "FiveDDataset", "FiveDDataset"]:
        """
        Split the dataset into train, validation, and test sets.
        
        Args:
            test_size: Proportion of data for the test set.
            val_size: Proportion of data for the validation set.
            random_state: Random seed for reproducibility.
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset).
            
        Raises:
            ValueError: If split proportions are invalid.
        """
        if test_size + val_size >= 1.0:
            raise ValueError(
                f"test_size ({test_size}) + val_size ({val_size}) "
                f"must be less than 1.0"
            )
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=random_state
        )
        
        # Second split: separate validation from training
        # Adjust val_size to account for the already removed test data
        adjusted_val_size = val_size / (1 - test_size)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=adjusted_val_size,
            random_state=random_state
        )
        
        # Create new dataset objects
        train_dataset = FiveDDataset(X_train, y_train)
        val_dataset = FiveDDataset(X_val, y_val)
        test_dataset = FiveDDataset(X_test, y_test)
        
        # Copy scaler reference if standardized
        if self.is_standardized and self.scaler is not None:
            train_dataset.scaler = self.scaler
            train_dataset.is_standardized = True
            val_dataset.scaler = self.scaler
            val_dataset.is_standardized = True
            test_dataset.scaler = self.scaler
            test_dataset.is_standardized = True
        
        return train_dataset, val_dataset, test_dataset
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.X.shape[0]
    
    def __repr__(self) -> str:
        """Return a string representation of the dataset."""
        return (
            f"FiveDDataset(n_samples={len(self)}, n_features=5, "
            f"standardized={self.is_standardized})"
        )


def load_dataset(filepath: Union[str, Path]) -> FiveDDataset:
    """
    Load a 5D dataset from a pickle file.
    
    The pickle file should contain either:
    - A dictionary with 'X' and 'y' keys
    - A tuple of (X, y)
    
    Args:
        filepath: Path to the .pkl file.
        
    Returns:
        FiveDDataset object containing the loaded data.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        DataValidationError: If the data format is invalid.
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    if filepath.suffix not in [".pkl", ".pickle"]:
        raise DataValidationError(
            f"Invalid file format. Expected .pkl or .pickle, got {filepath.suffix}"
        )
    
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    
    # Handle different data formats
    if isinstance(data, dict):
        if "X" not in data or "y" not in data:
            raise DataValidationError(
                "Dictionary must contain 'X' and 'y' keys. "
                f"Found keys: {list(data.keys())}"
            )
        X, y = data["X"], data["y"]
        
    elif isinstance(data, (tuple, list)) and len(data) == 2:
        X, y = data
        
    else:
        raise DataValidationError(
            "Invalid data format. Expected dict with 'X' and 'y' keys, "
            "or tuple/list of (X, y)"
        )
    
    return FiveDDataset(X, y)


def load_dataset_from_npz(filepath: Union[str, Path]) -> FiveDDataset:
    """
    Load a 5D dataset from a .npz file.
    
    The npz file should contain 'X' and 'y' arrays.
    
    Args:
        filepath: Path to the .npz file.
        
    Returns:
        FiveDDataset object containing the loaded data.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        DataValidationError: If the data format is invalid.
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    data = np.load(filepath)
    
    if "X" not in data or "y" not in data:
        raise DataValidationError(
            "NPZ file must contain 'X' and 'y' arrays. "
            f"Found arrays: {list(data.keys())}"
        )
    
    return FiveDDataset(data["X"], data["y"])


def create_sample_dataset(
    n_samples: int = 1000,
    noise_level: float = 0.1,
    random_state: Optional[int] = 42
) -> FiveDDataset:
    """
    Create a sample 5D dataset for testing purposes.
    
    Generates data where y is a nonlinear function of the 5 features.
    
    Args:
        n_samples: Number of samples to generate.
        noise_level: Standard deviation of Gaussian noise added to y.
        random_state: Random seed for reproducibility.
        
    Returns:
        FiveDDataset object containing the generated data.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate random features
    X = np.random.randn(n_samples, 5).astype(np.float32)
    
    # Create a nonlinear target function
    y = (
        2.0 * X[:, 0] +
        1.5 * X[:, 1] ** 2 +
        0.5 * X[:, 2] * X[:, 3] +
        np.sin(X[:, 4]) +
        noise_level * np.random.randn(n_samples)
    ).astype(np.float32)
    
    return FiveDDataset(X, y)