import numpy as np

def find_regressors(T: int, cutoff: float) -> np.ndarray:
    """
    This method creates q+1 trigonometric low-frequency regressors
    to project a time series of T elements onto.
    """
    V: np.ndarray = np.zeros((T,T))
    for i in range(T):
        for j in range(T):
            V[i][j] = min(i+1, j+1)
    R: np.ndarray = np.ones((T, 2))
    for t in range(T):
        R[t][1] = t+1-(T+1)/2    
    q: int = int(T*cutoff/2)
    inverse: np.ndarray = np.linalg.inv(np.matmul(R.T, R))
    M: np.ndarray = np.identity(T)-np.linalg.multi_dot([R, inverse, R.T])
    MVM: np.ndarray = np.linalg.multi_dot([M, V, M])
    regressors: list = []
    regressors.append(R[:, 0])
    regressors.append(R[:, 1])
    eigenvalues, eigenvectors = np.linalg.eig(MVM)
    indices: np.ndarray = np.argsort(eigenvalues)
    i: int = len(indices)-1
    while i >= 0 and len(regressors) < q+1:
        regressors.append(eigenvectors[indices[i]])
        i -= 1
    assert len(regressors) == q+1
    return np.array(regressors)

def find_time_series_regressors(y: np.array, cutoff: float) -> np.ndarray:
    """
    Finds the regressors for an unbalanced time series y
    """
    T: int = len(y)
    q: int = int(T*cutoff/2)

    V: np.ndarray = np.zeros((T,T))
    for i in range(T):
        for j in range(T):
            V[i][j] = min(i+1, j+1)

    R: np.ndarray = np.ones((T, 2))
    for t in range(T):
        R[t][1] = t+1-(T+1)/2    
    
    Ji: np.ndarray = np.zeros((T,T))
    for t in range(T):
        if (y[t] is not None):
            Ji[t][t] = 1

    Ri: np.ndarray = np.matmul(Ji, R)
    inverse: np.ndarray = np.linalg.inv(np.matmul(Ri.T, Ri))
    Mi: np.ndarray = Ji - np.linalg.multi_dot([Ri, inverse, Ri.T])
    MVM: np.ndarray = np.linalg.multi_dot([Mi, V, Mi])
    
    regressors: list = []
    regressors.append(R[:, 0])
    regressors.append(R[:, 1])
    
    eigenvalues, eigenvectors = np.linalg.eig(MVM)
    indices: np.ndarray = np.argsort(eigenvalues)
    
    i: int = len(indices)-1
    while i >= 0 and len(regressors) < q+1:
        regressors.append(eigenvectors[indices[i]])
        i -= 1
    assert len(regressors) == q+1
    return np.array(regressors)