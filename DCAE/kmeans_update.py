from sklearn.cluster import KMeans
import torch


def update_cluster_center(encoded_features):
    # Assuming encoded_features is a 2D tensor of shape (num_samples, num_features)
    # and we want to cluster these features into k clusters

    # Convert PyTorch tensor to numpy array if necessary
    encoded_features_np = encoded_features.cpu().detach().numpy()

    # Define the number of clusters
    num_clusters = 10  # for example, set to the number of classes in CIFAR-100

    # Initialize the KMeans algorithm with the desired number of clusters
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)

    # Fit the KMeans algorithm on the encoded features to compute cluster centers
    kmeans.fit(encoded_features_np)

    # Retrieve the cluster centers
    cluster_centers = kmeans.cluster_centers_

    # Convert cluster centers back to PyTorch tensor and send to the device
    cluster_centers_tensor = torch.tensor(cluster_centers).to(encoded_features.device)
    # Now you can use cluster_centers_tensor in your loss function
    return cluster_centers_tensor
