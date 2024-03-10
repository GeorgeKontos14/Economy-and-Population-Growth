import torch

def find_B(A, C, device):
    a = len(A)
    c = len(C)
    B = torch.zeros((a, c), device=device)
    for i in range(a):
        B[i, :] = A[i]*C
    return B.T

def make_positive_definite(matrix, epsilon=1e-6):
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix, UPLO='U')  # 'U' for upper triangular part
    # Clip eigenvalues to ensure they are positive or greater than epsilon
    clipped_eigenvalues = torch.clamp(eigenvalues, min=epsilon)
    # Reconstruct the matrix with modified eigenvalues
    modified_matrix = eigenvectors @ torch.diag(clipped_eigenvalues) @ eigenvectors.T
    return modified_matrix

def draw_independent_samples(mean, covariance, num_blocks, num_samples=1):
    # Get the size of each block
    block_size = mean.shape[0] // num_blocks
    
    # Reshape the mean vector to have shape (num_blocks, block_size)
    mean = mean.view(num_blocks, block_size)
    
    # Draw independent samples for each block
    samples = []
    for i in range(num_blocks):
        block_mean = mean[i]  # Extract mean for each block
        block_covariance = make_positive_definite(covariance[i * block_size : (i + 1) * block_size, i * block_size : (i + 1) * block_size])
        
        # Create a MultivariateNormal distribution for the block
        mvn_distribution = torch.distributions.MultivariateNormal(block_mean, block_covariance)
        
        # Draw samples from the distribution
        block_samples = mvn_distribution.sample((num_samples,))
        samples.append(block_samples)
    
    # Stack the samples along the last dimension
    samples = torch.cat(samples, dim=1)
    
    return samples[0]