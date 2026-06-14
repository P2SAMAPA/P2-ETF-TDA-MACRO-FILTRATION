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

def betti_numbers(dist_matrix, max_distance=0.6, max_dim=1):
    """
    Compute Betti numbers (β₀ and β₁) from a distance matrix using Rips complex.
    Fallback to graph method if gudhi is not available.
    """
    n = dist_matrix.shape[0]
    if n < 2:
        return 1, 0
    try:
        import gudhi as gd
        rips = gd.RipsComplex(distance_matrix=dist_matrix, max_edge_length=max_distance)
        st = rips.create_simplex_tree(max_dimension=max_dim)
        st.persistence()
        betti0 = st.persistent_betti_numbers(0, 0)[0]
        betti1 = st.persistent_betti_numbers(1, 0)[0] if max_dim >= 1 else 0
        return betti0, betti1
    except:
        # Fallback: use graph Laplacian to approximate
        adj = (dist_matrix < max_distance).astype(float)
        np.fill_diagonal(adj, 0)
        from scipy.sparse.csgraph import connected_components
        n_components, labels = connected_components(adj, directed=False)
        n_edges = np.sum(adj) / 2
        betti1 = max(0, n_edges - n + n_components)
        return n_components, betti1

def tda_macro_scores(returns, macro_df, base_max_distance=0.6):
    """
    Compute per‑ETF topological importance = node degree in macro‑adjusted Rips graph.
    """
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
    adj = (dist < scaled_max_distance).astype(int)
    # Node degree (number of close neighbours)
    degrees = np.sum(adj, axis=1)
    return degrees
