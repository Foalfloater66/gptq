import torch
from .abstractquant import QuantizerInterface
import warnings

class KMeansQuantizer(QuantizerInterface):
    """
    K-Means based quantizer. Finds k=2^bits centroids based on the input data
    and quantizes inputs to the nearest centroid.
    Currently implements a basic PyTorch K-Means for find_params.
    """

    def __init__(self, shape=1): # Shape not relevant here
        super(KMeansQuantizer, self).__init__()
        self.bits = -1
        self.k = -1
        self.register_buffer("centroids", torch.zeros(1)) 
        self.register_buffer('_is_ready', torch.tensor(False))
        self.max_kmeans_iter = 100 # Max iterations for K-Means
        self.kmeans_tol = 1e-4     # Tolerance for K-Means convergence

    def configure(self, bits, **kwargs):
        """Configure the quantizer with the number of bits."""
        if bits <= 0:
            raise ValueError("Number of bits must be positive.")
        self.bits = bits
        self.k = 2**bits

    def _kmeans_find_params(self, x):
        """Internal K-Means implementation."""
        dev = x.device
        x_flat = x.flatten().unsqueeze(1) # K-Means works on N x 1 data
        n_samples = x_flat.shape[0]

        if n_samples < self.k:
            warnings.warn(f"Number of samples ({n_samples}) is less than k ({self.k}). Using unique samples as centroids.")
            centroids = torch.unique(x_flat, dim=0)
            # Pad if still fewer than k unique samples
            if centroids.shape[0] < self.k:
                 padding = torch.zeros(self.k - centroids.shape[0], 1, device=dev, dtype=x.dtype)
                 centroids = torch.cat([centroids, padding], dim=0)
            self.centroids = centroids.squeeze(1) # Store as 1D tensor
            return

        indices = torch.randperm(n_samples, device=dev)[:self.k]
        centroids = x_flat[indices]
        
        # Iterative K-Means
        for i in range(self.max_kmeans_iter):
            # Assign points to nearest centroid
            # diffs shape will be (n_samples, k, 1) due to broadcasting
            diffs = torch.abs(x_flat.unsqueeze(1) - centroids.unsqueeze(0))
            # assignments should be 1D [n_samples], use keepdim=False
            assignments = torch.argmin(diffs, dim=1, keepdim=False) # Shape: [n_samples]

            # Store old centroids for convergence check
            old_centroids = centroids.clone()

            # Update centroids: mean of assigned points
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(self.k, device=dev)

            # Efficiently update centroids using scatter_add_
            # 'assignments' has shape [N, 1] after argmin on dim=1 of [N, k, 1]
            # 'x_flat' has shape [N, 1]
            # 'new_centroids' has shape [k, 1]
            # We scatter along dim 0 using 'assignments' as index and 'x_flat' as source.
            # The index tensor 'assignments' must have the same number of dimensions as the src 'x_flat'.
            new_centroids.scatter_add_(0, assignments, x_flat)
            
            # For counts (shape [k]), the index needs to be 1D [N]. Squeeze assignments.
            # The source 'ones' also needs to be 1D [N].
            ones_for_counts = torch.ones(assignments.shape[0], device=dev, dtype=torch.float)
            counts.scatter_add_(0, assignments.squeeze(1), ones_for_counts)

            # Avoid division by zero for empty clusters - keep old centroid
            mask_empty = counts == 0
            counts[mask_empty] = 1.0 # Prevent NaN
            new_centroids /= counts.unsqueeze(1)
            
            # Keep old centroid if cluster became empty
            new_centroids[mask_empty.squeeze()] = old_centroids[mask_empty.squeeze()] 
            
            centroids = new_centroids

            # Check for convergence
            centroid_shift = torch.norm(centroids - old_centroids)
            if centroid_shift < self.kmeans_tol:
                # print(f"K-Means converged in {i+1} iterations.")
                break

        self.centroids = torch.sort(centroids.squeeze(1))[0]


    def find_params(self, x, weight=False, **kwargs):
        """
        Runs K-Means on the input data x to find k=2^bits centroids.
        """
        if self.k <= 0:
            raise ValueError("KMeansQuantizer not configured. Call configure() first.")

        dev = x.device
        self.centroids = self.centroids.to(dev)
        self._is_ready = self._is_ready.to(dev)

        # Run K-Means
        self._kmeans_find_params(x.float()) 

        self._is_ready.copy_(torch.tensor(True))
        return

    def quantize(self, x):
        """
        Quantizes input tensor x by mapping each element to the nearest centroid.
        """
        if not self.ready():
            warnings.warn("KMeansQuantizer not ready, returning original tensor.")
            return x

        centroids = self.centroids.to(x.device)
        original_shape = x.shape
        x_flat = x.flatten()

        # Find nearest centroid for each point
        diffs = torch.abs(x_flat.unsqueeze(1) - centroids.unsqueeze(0)) # Shape: (N, k)
        assignments = torch.argmin(diffs, dim=1) # Shape: (N)

        # Map to centroids
        q_flat = centroids[assignments]

        return q_flat.view(original_shape).to(x.dtype)

    def ready(self):
        """Checks if the quantizer is ready (centroids found)."""
        return self.k > 0 and self._is_ready.item()
