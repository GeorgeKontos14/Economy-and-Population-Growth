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

def update_U_c_i(U_c, X_i, F, mu_c, lambdas, G, J, q_hat, device):
    new_U = []
    i_1 = torch.zeros(q_hat+1, device=device)
    i_1[0] = 0
    for i in range(len(U_c)):
        new_U.append(X_i[i]-F-mu_c*i_1-lambdas.lambda_c_i[i]*G[J[i]])
    return new_U

def update_U_g_j(U_g, G, lambdas, K, H):
    new_U = []
    for i in range(len(U_g)):
        new_U.append(G[i]-lambdas.lambda_g_j[i]*H[K[i]])
    return new_U