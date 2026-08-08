import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional, Callable
import matplotlib.pyplot as plt
import warnings
pic_name = "target_distribution"

class SMOTER:
    """
    SMOTE for Regression (SMOTER) Implementation
    
    This class implements the SMOTE algorithm adapted for regression tasks.
    It generates synthetic samples for rare/important regions of the target variable.
    
    Parameters:
    -----------
    k : int, default=5
        Number of nearest neighbors to consider
    relevance_threshold : float, default=0.5
        Threshold above which samples are considered rare (important)
    oversampling_ratio : float or str, default='balance'
        Oversampling ratio. If 'balance', automatically balances the dataset
    random_state : int, default=None
        For reproducible results
    relevance_function : callable, default=None
        Custom relevance function. If None, automatic method is used
    """
    
    def __init__(
        self,
        k: int = 5,
        relevance_threshold: float = 0.5,
        oversampling_ratio: float = 'balance',
        random_state: Optional[int] = None,
        relevance_function: Optional[Callable] = None
    ):
        self.k = k
        self.relevance_threshold = relevance_threshold
        self.oversampling_ratio = oversampling_ratio
        self.random_state = random_state
        self.relevance_function = relevance_function
        self._scaler = None
        
        # Set random seed for reproducibility
        if random_state is not None:
            np.random.seed(random_state)
    
    def _calculate_relevance_auto(self, y: np.ndarray) -> np.ndarray:
        """
        Automatic relevance calculation based on data distribution
        
        Samples at the tails of the distribution (extreme values) get higher relevance.
        Uses IQR (Interquartile Range) to define the core region.
        
        Parameters:
        -----------
        y : np.ndarray
            Target values
            
        Returns:
        --------
        relevance : np.ndarray
            Relevance scores between 0 and 1 for each sample
        """
        # Calculate percentiles
        q1 = np.percentile(y, 25)
        q3 = np.percentile(y, 75)
        iqr = q3 - q1
        
        # Define the middle region (low relevance zone)
        lower_bound = q1 - 0.5 * iqr
        upper_bound = q3 + 0.5 * iqr
        
        # Initialize relevance array
        relevance = np.zeros_like(y, dtype=float)
        
        for i, val in enumerate(y):
            if lower_bound <= val <= upper_bound:
                # Samples in the middle region: low relevance
                relevance[i] = 0.0
            elif val < lower_bound:
                # Low values: high relevance (near 1)
                # Normalize based on distance from lower bound
                min_val = y.min()
                range_val = lower_bound - min_val
                if range_val > 0:
                    relevance[i] = min(1.0, (lower_bound - val) / range_val)
                else:
                    relevance[i] = 1.0
            else:  # val > upper_bound
                # High values: high relevance (near 1)
                max_val = y.max()
                range_val = max_val - upper_bound
                if range_val > 0:
                    relevance[i] = min(1.0, (val - upper_bound) / range_val)
                else:
                    relevance[i] = 1.0
        
        return relevance
    
    def _calculate_relevance_custom(self, y: np.ndarray) -> np.ndarray:
        """
        Use a custom user-defined relevance function
        
        Parameters:
        -----------
        y : np.ndarray
            Target values
            
        Returns:
        --------
        relevance : np.ndarray
            Relevance scores from custom function
        """
        if self.relevance_function is None:
            raise ValueError("Custom relevance function is not defined")
        return self.relevance_function(y)
    
    def _get_relevance(self, y: np.ndarray) -> np.ndarray:
        """
        Calculate relevance scores using the appropriate method
        
        Parameters:
        -----------
        y : np.ndarray
            Target values
            
        Returns:
        --------
        relevance : np.ndarray
            Relevance scores for each sample
        """
        if self.relevance_function is not None:
            return self._calculate_relevance_custom(y)
        else:
            return self._calculate_relevance_auto(y)
    
    def _get_rare_indices(self, relevance: np.ndarray) -> np.ndarray:
        """
        Identify rare/important samples based on relevance threshold
        
        Parameters:
        -----------
        relevance : np.ndarray
            Relevance scores
            
        Returns:
        --------
        rare_indices : np.ndarray
            Indices of samples with relevance > threshold
        """
        return np.where(relevance > self.relevance_threshold)[0]
    
    def _calculate_oversampling_amount(self, y: np.ndarray, rare_indices: np.ndarray) -> int:
        """
        Calculate how many synthetic samples need to be generated
        
        Parameters:
        -----------
        y : np.ndarray
            Target values
        rare_indices : np.ndarray
            Indices of rare samples
            
        Returns:
        --------
        n_samples_to_generate : int
            Number of synthetic samples to create
        """
        n_samples = len(y)
        n_rare = len(rare_indices)
        
        # No rare samples found
        if n_rare == 0:
            warnings.warn("No rare samples found! Dataset is already balanced.")
            return 0
        
        # Calculate oversampling amount
        if self.oversampling_ratio == 'balance':
            # Goal: make rare samples count equal to normal samples count
            n_normal = n_samples - n_rare
            target_n_rare = n_normal
            oversample_count = target_n_rare - n_rare
            return max(0, int(oversample_count))
        else:
            # Generate n_rare * oversampling_ratio new samples
            return int(n_rare * self.oversampling_ratio)
    
    def _find_neighbors(self, X: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """
        Find k-nearest neighbors for selected samples
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        indices : np.ndarray
            Indices of samples to find neighbors for
            
        Returns:
        --------
        neighbor_indices : np.ndarray
            Indices of k neighbors for each sample (excluding itself)
        """
        # Fit nearest neighbors model
        nbrs = NearestNeighbors(n_neighbors=min(self.k + 1, len(X)), metric='euclidean')
        nbrs.fit(X)
        
        # Find distances and indices of neighbors
        distances, neighbor_indices = nbrs.kneighbors(X[indices])
        
        # Remove the first column (the sample itself)
        # First neighbor is always the sample itself
        return neighbor_indices[:, 1:]  # Return k neighbors (excluding self)
    
    def _generate_synthetic_samples(
        self,
        X: np.ndarray,
        y: np.ndarray,
        rare_indices: np.ndarray,
        n_samples_to_generate: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic samples using interpolation between rare samples and their neighbors
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        rare_indices : np.ndarray
            Indices of rare samples
        n_samples_to_generate : int
            Total number of synthetic samples to generate
            
        Returns:
        --------
        synthetic_X : np.ndarray
            Generated synthetic features
        synthetic_y : np.ndarray
            Generated synthetic target values
        """
        if n_samples_to_generate <= 0:
            return np.array([]), np.array([])
        
        # Find neighbors for rare samples
        neighbor_indices = self._find_neighbors(X, rare_indices)
        
        synthetic_X = []
        synthetic_y = []
        
        # Calculate how many samples to generate from each rare sample
        n_rare = len(rare_indices)
        samples_per_point = n_samples_to_generate // n_rare
        remainder = n_samples_to_generate % n_rare
        
        for i, rare_idx in enumerate(rare_indices):
            # Number of samples to generate from this rare point
            n_samples = samples_per_point + (1 if i < remainder else 0)
            
            if n_samples == 0:
                continue
            
            # Get neighbors for this rare sample
            neighbors = neighbor_indices[i]
            
            # Randomly select neighbors (with replacement if needed)
            chosen_neighbors = np.random.choice(
                neighbors, 
                size=n_samples, 
                replace=(n_samples > len(neighbors))
            )
            
            # Generate synthetic samples through interpolation
            for neighbor_idx in chosen_neighbors:
                # Calculate difference between rare sample and its neighbor
                diff = X[neighbor_idx] - X[rare_idx]
                
                # Random interpolation factor between 0 and 1
                gap = np.random.random()
                
                # Create synthetic sample: rare_sample + gap * difference
                synthetic_point = X[rare_idx] + gap * diff
                synthetic_X.append(synthetic_point)
                
                # Target value: weighted average based on distance
                # (Alternative: simple average of the two target values)
                synthetic_y.append((y[rare_idx] + y[neighbor_idx]) / 2)
        
        return np.array(synthetic_X), np.array(synthetic_y)
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTER algorithm to resample the dataset
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features
        y : array-like, shape (n_samples,)
            Target variable for regression
            
        Returns:
        --------
        X_resampled : array-like, shape (n_samples + n_synthetic, n_features)
            Original features + synthetic features
        y_resampled : array-like, shape (n_samples + n_synthetic,)
            Original target + synthetic target values
        """
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y).flatten()
        
        # Normalize features for better distance calculation
        self._scaler = MinMaxScaler()
        X_scaled = self._scaler.fit_transform(X)
        
        # Calculate relevance scores for each sample
        relevance = self._get_relevance(y)
        
        # Identify rare/important samples
        rare_indices = self._get_rare_indices(relevance)
        
        # If no rare samples found, return original data
        if len(rare_indices) == 0:
            warnings.warn("No rare samples found! Returning original data.")
            return X, y
        
        # Calculate how many synthetic samples to generate
        n_samples_to_generate = self._calculate_oversampling_amount(y, rare_indices)
        
        if n_samples_to_generate == 0:
            warnings.warn("Number of samples to generate is 0! Returning original data.")
            return X, y
        
        # Generate synthetic samples
        synthetic_X_scaled, synthetic_y = self._generate_synthetic_samples(
            X_scaled, y, rare_indices, n_samples_to_generate
        )
        
        if len(synthetic_X_scaled) == 0:
            return X, y
        
        # Scale synthetic features back to original scale
        synthetic_X = self._scaler.inverse_transform(synthetic_X_scaled)
        
        # Combine original and synthetic data
        X_resampled = np.vstack([X, synthetic_X])
        y_resampled = np.concatenate([y, synthetic_y])
        
        return X_resampled, y_resampled
    
    def fit_resample_dataframe(
        self, 
        df: pd.DataFrame, 
        target_column: str
    ) -> pd.DataFrame:
        """
        Convenience method to work with pandas DataFrames
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing features and target
        target_column : str
            Name of the target column
            
        Returns:
        --------
        df_resampled : pandas.DataFrame
            New DataFrame with original + synthetic data
        """
        # Separate features and target
        X = df.drop(columns=[target_column]).values
        y = df[target_column].values
        
        # Apply SMOTER
        X_resampled, y_resampled = self.fit_resample(X, y)
        
        # Reconstruct DataFrame
        feature_columns = df.columns.drop(target_column)
        df_resampled = pd.DataFrame(X_resampled, columns=feature_columns)
        df_resampled[target_column] = y_resampled
        
        return df_resampled


# ============================================
# Main function
# ============================================
def main(df: pd.DataFrame, target_column: str):

    # 1. split dataframe
    y = df[target_column]
    X = df.drop(columns=target_column)

    print(f"Original dataset size: {len(df)}")
    print(f"Target distribution: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")
    print(f"Standard deviation: {y.std():.2f}")
    print("\n" + "="*50 + "\n")

    # 2. Apply SMOTER
    smoter = SMOTER(
        k=5,
        relevance_threshold=0.6,
        oversampling_ratio='balance',
        random_state=42
    )

    df_resampled = smoter.fit_resample_dataframe(df, target_column)

    print(f"Resampled dataset size: {len(df_resampled)}")
    print(f"Added samples: {len(df_resampled) - len(df)}")
    
    y_resampled = df_resampled[target_column].values
    print(f"New target distribution: min={y_resampled.min():.2f}, max={y_resampled.max():.2f}, mean={y_resampled.mean():.2f}")
    print(f"New standard deviation: {y_resampled.std():.2f}")

    # 3. Visualize the distribution difference
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.hist(y, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_title('Target Distribution BEFORE SMOTER')
    ax1.set_xlabel('Target Value')
    ax1.set_ylabel('Frequency')
    
    ax2.hist(y_resampled, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax2.set_title('Target Distribution AFTER SMOTER')
    ax2.set_xlabel('Target Value')
    ax2.set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f'Figures/{pic_name}.png', dpi=300, bbox_inches='tight')
    print(f'{pic_name}.png')

if __name__ == "__main__":
    # Example
    # 1. Create imbalanced dataset (with rare values at both extremes)
    np.random.seed(42)
    
    # Normal samples (majority)
    n_normal = 800
    X_normal = np.random.randn(n_normal, 3) * 0.5 + 1
    y_normal = np.random.normal(10, 2, n_normal)
    
    # Rare low values (minority)
    n_rare_low = 50
    X_rare_low = np.random.randn(n_rare_low, 3) * 0.3 - 2
    y_rare_low = np.random.normal(2, 0.5, n_rare_low)
    
    # Rare high values (minority)
    n_rare_high = 50
    X_rare_high = np.random.randn(n_rare_high, 3) * 0.3 + 3
    y_rare_high = np.random.normal(18, 0.5, n_rare_high)
    
    # Combine all samples
    X = np.vstack([X_normal, X_rare_low, X_rare_high])
    y = np.concatenate([y_normal, y_rare_low, y_rare_high])
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=['feature_1', 'feature_2', 'feature_3'])
    df['target'] = y
    
    print(f"Original dataset size: {len(df)}")
    print(f"Target distribution: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")
    print(f"Standard deviation: {y.std():.2f}")
    print("\n" + "="*50 + "\n")
    
    main(df= df, target_column= 'target')