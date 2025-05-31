def annotation(adata,train_index,test_index):
    train_data = adata.obsm["Embedding"][train_index]
    test_data = adata.obsm["Embedding"][test_index]

    train_label = adata.obs["MajorType"][train_index]
    test_label = adata.obs["MajorType"][test_index]

    test_adata = adata[test_index].copy()
    transfer_matrix = test_adata.obsm["Similarity"][:,train_index]
    
    k = 10
    similarity_matrix = transfer_matrix
    n_targets, n_sources = similarity_matrix.shape
    knn_indices = np.zeros((n_targets, k), dtype=int)
    knn_distances = np.zeros((n_targets, k))
    epsilon=1e-6
    
    for i in range(n_targets):
            similarities = similarity_matrix[i, :]
            partitioned_indices = np.argpartition(-similarities, k)[:k]
            sorted_indices = partitioned_indices[np.argsort(-similarities[partitioned_indices])]
            knn_indices[i, :] = sorted_indices
            knn_distances[i, :] = 1 / (similarities[sorted_indices] + epsilon)
            
            
    predict = np.zeros((n_targets, 1), dtype=int)
    for i in range(n_targets):
        labels = train_label[knn_indices[i]]
        weights = 1 / (knn_distances[i] + epsilon)
        weighted_votes = np.bincount(labels, weights=weights)
        predict[i, :] = np.argmax(weighted_votes)
    return predict