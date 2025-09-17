import numpy as np
from scipy.ndimage import gaussian_filter1d
import igraph as ig

def cal_adjacency_mat(transition_mat,ds_scale=1,bins=400,sigma=2):
    adjacency_mat = np.zeros_like(transition_mat[::ds_scale,::ds_scale])
    
    for i in range(len(adjacency_mat)):
        tostudy_idx = np.arange(len(transition_mat))
        mat_idx = np.delete(tostudy_idx,i*ds_scale)
        freqs = np.histogram(transition_mat[::ds_scale][i][mat_idx],bins=bins)
        smoothed_freqs = gaussian_filter1d(freqs[0], sigma=sigma)

        try:
            grad = np.gradient(smoothed_freqs, np.arange(len(smoothed_freqs)))
            left_value2 = grad[:-4] < 0
            left_value1 = grad[1:-3] <= 0

            right_value1 = grad[2:-2] > 0
            right_value2 = grad[3:-1] > 0
            judge_condition1 = left_value2*left_value1*right_value1*right_value2

            left_value2 = grad[:-4] < 0
            left_value1 = grad[1:-3] < 0

            right_value1 = grad[2:-2] >= 0
            right_value2 = grad[3:-1] > 0
            judge_condition2 = left_value2*left_value1*right_value1*right_value2

            threshold_idx = np.where(judge_condition1+judge_condition2)[0].max()
            x_distribution = (freqs[1][:-1] + freqs[1][1:])/2
            dx_distribution = (x_distribution[:-1] + x_distribution[1:])/2
            threshold = (dx_distribution[threshold_idx+1] + dx_distribution[threshold_idx+2])/2
        except:
            threshold = -np.inf
        adjacency_mat[i][transition_mat[::ds_scale,::ds_scale][i]<threshold] = 1

    adjacency_mat += adjacency_mat.T
    adjacency_mat = np.int64(adjacency_mat>1)
    np.fill_diagonal(adjacency_mat,0)
    return adjacency_mat

def find_clique_idx(adjacency_mat):
    num_vertices = len(adjacency_mat)
    edges_mat = np.triu(adjacency_mat, k=0)
    edges_ = np.where(edges_mat)
    edges = [(i,j) for i,j in zip(*edges_)]

    g = ig.Graph(n=num_vertices, edges=edges, directed=False)

    largest_cliques = g.largest_cliques()
    if largest_cliques:
        max_clique = largest_cliques[0]
    clique_idx = np.array([*max_clique])
    return clique_idx
    
    
def self_consistent_assignment(label, transition_mat, tolerance=5, min_cluster_size=100, patience=15,change_patient_tolerence=10,
                               _previous_change_count=-1, _unchanged_streak=0):
    if not isinstance(label, np.ndarray):
        label = np.array(label)

    if label.size == 0:
        print("Input label array is empty. Returning empty label array.")
        return np.array([]) # Return an empty numpy array consistent with type

    unique_labels_for_centroids = np.unique(label)
    
    try:
        pxcy_avg = np.hstack([transition_mat[:, label == i].mean(axis=-1, keepdims=True)
                              for i in unique_labels_for_centroids])
    except ValueError as e:
        if "need at least one array to stack" in str(e) and unique_labels_for_centroids.size > 0:
            print(f"Error during np.hstack, possibly due to mat dimensions or empty slices despite unique labels: {e}")
            print(f"Mat shape: {transition_mat.shape}, unique_labels_for_centroids: {unique_labels_for_centroids}")
            for i in unique_labels_for_centroids:
                print(f"  For label {i}, slice shape: {transition_mat[:, label == i].shape}")
            return label # Cannot proceed
        raise e
        
    label_new = pxcy_avg.argmax(axis=1)
    
    change_count = (label != label_new).sum()
    print('the number of changed samples is %d' % (change_count))

    # --- Patience Logic ---
    current_unchanged_streak = _unchanged_streak
    if abs(change_count-_previous_change_count)<change_patient_tolerence:
        current_unchanged_streak += 1
    else:
        current_unchanged_streak = 0  # Reset streak if change_count changed

    # Check patience condition (only if patience is a positive value enabling the feature)
    if patience > 0 and current_unchanged_streak >= patience:
        print(f'Patience ({patience}) reached. Change count ({change_count}) '
              f'changed slightly for {current_unchanged_streak} rounds.')
        indice_not_sure = np.where(label != label_new)[0]
        return label_new,indice_not_sure  # Return the current (re-indexed) label_new
    # --- End of Patience Logic ---
    
    if change_count > tolerance:
        # `label_new` (the re-indexed labels) is used for further processing
        # and becomes the `label` for the next recursive call.
        labels_for_next_step = label_new.copy()

        unique_new_labels, counts_new = np.unique(labels_for_next_step, return_counts=True)
        small_size_cluster_label_indices = np.where(counts_new < min_cluster_size)[0]
        small_size_cluster_label = unique_new_labels[small_size_cluster_label_indices]

        if len(small_size_cluster_label_indices) == 0:
            # No small clusters to remove, recurse
            return self_consistent_assignment(labels_for_next_step, transition_mat, tolerance, min_cluster_size,
                                              patience,change_patient_tolerence, change_count, current_unchanged_streak)
        else:
            # Small clusters found, get their actual label values (which are canonical 0-indexed)
            canonical_labels_to_reset = unique_new_labels[small_size_cluster_label_indices]
            for c_label_to_reset in canonical_labels_to_reset:
                # set transition probability of small cluster to -inf
                pxcy_avg_small_size_cluster = pxcy_avg[labels_for_next_step == c_label_to_reset]
                pxcy_avg_small_size_cluster[:,small_size_cluster_label] = -np.inf
                
                labels_for_next_step[labels_for_next_step == c_label_to_reset] = pxcy_avg_small_size_cluster.argsort(axis=1)[:,-1]
                

            return self_consistent_assignment(labels_for_next_step, transition_mat, tolerance, min_cluster_size,
                                              patience,change_patient_tolerence, change_count, current_unchanged_streak)
    else:  # change_count <= tolerance
        print(f'Change count ({change_count}) is within tolerance ({tolerance}).')
        return label_new  # Return the current (re-indexed) label_new
    
def merging(transition_mat, ds_scale=1, bins=400, sigma=2, tolerance=5, min_cluster_size=100, patience=15,change_patient_tolerence=10):
    print("calculate adjaceny matrix ...")
    adjacency_mat = cal_adjacency_mat(transition_mat,ds_scale=ds_scale,bins=bins,sigma=sigma)

    print("find the largest clique ...")
    clique_idx = find_clique_idx(adjacency_mat)

    print("self-consistent reassignment ...")
    label_clique = transition_mat[:,clique_idx*ds_scale].argmax(1)
    merging_result = self_consistent_assignment(label_clique,transition_mat,tolerance=tolerance,min_cluster_size=min_cluster_size,\
                                            patience=patience,change_patient_tolerence=change_patient_tolerence)
    return merging_result