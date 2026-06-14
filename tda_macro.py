import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

def compute_composite_macro_factor(macro_df):
    """
    Compute a composite macro factor as the first principal component of all macro variables.
    Returns array of length len(macro_df).
    """
    if macro_df is None or len(macro_df) < 2:
        return np.ones(len(macro_df)) * 0.5
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    pca = PCA(n_components=1)
    pca.fit(macro_scaled)
    factor = pca.transform(macro_scaled).flatten()
    # Normalise to [0,1] range
    factor = (factor - factor.min()) / (factor.max() - factor.min() + 1e-8)
    return factor

def eigenvector_centrality(adj, max_iter=100, tol=1e-6):
    """
    Compute eigenvector centrality (power iteration).
    """
    n = adj.shape[0]
    # Initial guess
    v = np.ones(n)
    for _ in range(max_iter):
        v_new = adj.T @ v
        v_new = v_new / (np.linalg.norm(v_new) + 1e-12)
        if np.linalg.norm(v_new - v) < tol:
            break
        v = v_new
    return v

def tda_macro_scores(returns, macro_df, base_max_distance=0.6):
    """
    Compute per‑ETF topological importance = eigenvector centrality in macro‑adjusted Rips graph.
    """
    n = returns.shape[1]
    if n < 2:
        return np.zeros(n)
    # Build distance matrix from correlation distance
    corr = returns.corr().values
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    # Compute composite macro factor (last value)
    if macro_df is not None and len(macro_df) > 0:
        macro_factor = compute_composite_macro_factor(macro_df)
        current_macro_factor = macro_factor[-1]
    else:
        current_macro_factor = 0.5
    # Adjust max distance based on macro factor
    scaled_max_distance = base_max_distance * (1 - current_macro_factor * 0.5)
    scaled_max_distance = max(0.1, min(0.9, scaled_max_distance))
    # Build adjacency matrix (edges where distance < scaled threshold)
    adj = (dist < scaled_max_distance).astype(float)
    # Ensure symmetry
    adj = np.maximum(adj, adj.T)
    # Compute eigenvector centrality
    centrality = eigenvector_centrality(adj)
    return centrality
