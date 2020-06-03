import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier


#* Class for data augmentation in binary classification
class EigenSampler():
    
    
    def __init__(self, X, y, treshold = .9, C = 10, epsilon = 0.01, scale = True):
        # Data attributes
        self.X_std = X
        self.y = y
        if scale:
            self.X_std = StandardScaler().fit_transform(X)
            self.mean_vec = np.mean(self.X_std, axis=0)
        self.mean_vec = np.mean(self.X_std, axis=0) 
        # Data dimensions
        self.n, self.d = self.X_std.shape
        # Hyperparameters
        self.treshold = treshold
        self.C = C
        self.epsilon = epsilon
        self.n_cluster = np.sqrt(self.n / 2)
        # Flow control
        self.has_fit = False
        self.has_generate = False
        self.has_classify = False
       
    
    def projection_matrix(self, eig_vectors, k):
        # return the projection matrix to the reduced data space
        # the matrix is the eigenvectors as columns (since they already are orthonormal, cov_matrix is Hermitian)
        return eig_vectors[:k].T
    
    
    def new_dimension(self, eig_values):
        # Finds the dimension of the new reduced dataspace used in the following.
        # We choose k in such a manner that the ratio of sum of squares of the selected eigenvalues over the sum of
        # squares of all eigenvalues exceeds a chosen threshold.
        # This threshold is essentially a measure of the information we wish to preserve,
        # and is typically chosen to be greater than 90% (default is 90%).
        k = 1
        norm = np.sum(eig_values ** 2)
        squared_eig = eig_values ** 2
        ratio = np.sum(squared_eig[:k]) / norm
        while ratio <= self.treshold:
            k += 1
            ratio = np.sum(squared_eig[:k]) / norm
        
        return k
    
    
    def cluster(self):
        # Perform a cluster analysis of the smaller dataset (input)
        # return the clusters centers (list of points) and the cluster class for each data-point
        k_means = KMeans(n_clusters = int(self.n_cluster))
        k_means.fit(self.proj_X)
        # array of centers
        centroids = k_means.cluster_centers_
        # cluster index
        centers = k_means.labels_
        
        return centroids, centers
    
    
    def fit(self):
        # Get the necessary operators and parameters:
        # k: dimension of the smaller subspace
        # eig_vals and eig_vec: eigenvalues and eigenvectors that span the subspace found
        # performs a cluster analysis in the projected data in the reduced space
        # centroids: vector with the clusters centers, centers: map each projected data in one cluster
        B = self.X_std - self.mean_vec
        cov_mat = np.cov(B.T)
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)
        
        self.k = self.new_dimension(eig_vals)
        self.G = self.projection_matrix(eig_vecs, self.k)
        self.proj_X = self.X_std @ self.G
        self.centroids, self.centers = self.cluster()
        self.has_fit = True
        
        return
    
    
    def generate(self, get = False):
        # Obtain synthetic points: the middle point of a projected data-point and
        # the given center of the correspondent cluster it belongs to
        # return the augmented dataset in the reduced  dimension
        if not self.has_fit:
            return "First we must fit the parameters"
        new_points = []
        for index, old_point in enumerate(self.proj_X):
            point = (old_point + self.centroids[self.centers[index]])/2
            new_points.append(point)
        self.new_points = np.array(new_points)
        self.has_generate = True
        
        if get:
            return self.new_points
        
        return
    
    
    def classify(self, model):
        # Train the classifier in the old_points with the corresponding label as a supervised task
        # predict each new point class label with the trained classifier
        # return the vector with predicted labels
        if not self.has_generate:
            return "First we need to generate the synthetic data"
        model.fit(self.proj_X, self.y)
        self.label = model.predict(self.new_points)
        self.has_classify = True
        
        return self.label
    
    
    def reverse_projection(self):
        # Computes a suitable transformation of the new generated data points in the original data-space
        # aThis map permits a trade-off between
        # the norm of the solution ||z_i|| and the approximation error, while
        # also preserving the lower and upper bound constraints determined by the original dataset
        # The parameters lambda, mu, gamma and delta are the Lagrange multipliers used to solve this optimization task
        if not self.has_classify:
            print("You haven't classified the new data points!")
            
        tmp1 = self.G.T @ self.G
        tmp2 = -(tmp1 + (1/np.abs(self.C)) * np.eye(self.k))
        A = np.concatenate((tmp2, tmp1, self.G.T, -self.G.T), axis = 1)
        tmp3 = np.concatenate((tmp1, tmp2, -self.G.T, self.G.T), axis=1)
        A = np.concatenate((A, tmp3))
        
        inv_A = np.linalg.pinv(A)
        one = self.epsilon * np.ones(self.k).T
        high_dim_points = []
        
        for x_i in self.new_points:
            aux = np.concatenate((x_i + self.epsilon * one, -x_i + self.epsilon * one))
            solution = inv_A @ aux
            lambda_ , mu, gamma, delta = solution[:self.k], solution[self.k : 2 * self.k], solution[2 * self.k : 2 * self.k + self.d], solution[2 * self.k + self.d:]
            z_i = self.G @ (mu - lambda_) + (gamma - delta)
            high_dim_points.append(z_i.T)
        # re-scale the new data points to original center  
        self.high_dim_points = np.array(high_dim_points) + self.mean_vec
        #TODO: Check complex values
        
        return self.high_dim_points


def EigenSamplerClassifier(X, y, model, t = 0.001, cte = 0.1, report = False):
    #* This function performs an automatization of the class above.
    #* "t" and "cte" are hyperparameters, the values are the same used in the article
    #* "model" is expected to ba a sklearn classification model.
    #* It returns the new data points and theirs corresponding labels wrt the data [X:y]
    
    sampler = EigenSampler(X, y,treshold=t, C=cte)
    sampler.fit()
    sampler.generate()
    new_labels = sampler.classify(model)
    if report:
        print(f"EigenSample - Fraction of new labels: {np.sum(new_labels) / len(new_labels)}")
    new_data_reversed = sampler.reverse_projection()
    
    return new_data_reversed, new_labels


def DataLoader_Classification(path):
    data = pd.read_csv(path)
    X = data.drop("Y", axis = 1)
    y = data[["Y"]] 
    y = y.astype(int)
    
    return X, y


if __name__ == "__main__":
    '''
    Example:
    X, y = DataLoader_Classification("../data/data_final.csv")
    model = RandomForestClassifier()
    new_X, new_y = EigenSamplerClassifier(X, y.values, model)
    '''
    pass