import pandas as pd
import numpy as np
from EigenSampleClassifier import EigenSamplerClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


#* Class for data augmentation in regression framework
#* This methodds transform the problem into a classification framework
#* before calculating the continuous variables again. The performance compared to the
#* classification procedure may be less satisfactory.
class EigenSampler_Regressor():
    
    
    def __init__(self, X, y, scale = True):
        # Data attributes
        self.X_std = X
        self.y = y
        if scale:
            self.X_std = StandardScaler().fit_transform(X) 
        # Data dimensions
        self.n, self.d = self.X_std.shape
        # Flow control
        self.make_classification = False
    
    
    def fit(self, epsilon = 0.01):
        # new covariate
        self.y_tmp = np.concatenate((np.array(self.y) + epsilon, np.array(self.y) - epsilon))
        # respective classes
        self.y_class = np.concatenate((np.ones(int(self.n)), np.zeros(int(self.n))))
        self.y_tmp = self.y_tmp.reshape(2 * self.n, 1)
        # new X matrix
        self.X_aug = np.concatenate((self.X_std, self.X_std))
        self.X_aug = np.concatenate((self.X_aug, self.y_tmp), axis=1)
        self.mean_vec = np.mean(self.y_tmp)
        self.X_aug = StandardScaler().fit_transform(self.X_aug)
        
        self.make_classification = True
        return
    
    
    def data_augmentation(self, model):
        if not self.make_classification:
            return "Transform data to a suitable format"
        x_new, _ = EigenSamplerClassifier(self.X_aug, self.y_class, model)
        
        return x_new[:, :-1], x_new[:,-1] + self.mean_vec


def EigenSamplerRegressor(X, y, classifier = SVC(C = .5, kernel='linear'), epsilon = 0.01):
    #* This function performs an automatization of the class above.
    #* epsilon is a hyperparameters, one is free to change. "classifier" is expected to be a sklearn
    #* classifier funcation.
    #* The SVC is the standard classifier to perform the classification in the reduced data spaced before turning the
    #* problem to a regression task. We rely on the SCV's ability
    #* do adapt to the decision boundary with good flexibility.
    #* It returns the new data points and theirs corresponding targets

    sampler = EigenSampler_Regressor(X, y)
    sampler.fit(epsilon=epsilon)
    X_new, y_new = sampler.data_augmentation(model=classifier)

    return np.array(X_new), np.array(y_new)


def DataLoader_Regresion(path, year = 2010):
    data = pd.read_csv(path)
    X = data.drop("Y", axis = 1) # Y is our target variable, one may change as they wish
    y = data[["Y"]]
        
    return X, y


if __name__ == "__main__":
    '''
    X_train, y_train = DataLoader_Regresion("../data/data_final.csv")
    X_new, y_new = EigenSamplerRegressor(X_train, y_train)
    print(f"Shape of new_X: {X_new.shape}")
    print(f"Shape of new_y: {y_new.shape}")
    print(y_new)
    '''
    pass
    
        
        