import numpy as np
import time

# EVALUATION METRICS

# Precision (how many of the recommended items are relevant):
def precision(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True) # Test whether each element of a 1-D array is also present in a second array
    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)
    return precision_score


# Recall (how many of the relevant items I was able to recommend):
def recall(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]
    return recall_score


# Mean Average Precision:
def MAP(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    # Cumulative sum: precision at 1, at 2, at 3 ...
    # np.cumsum(...): each element at position i represents the cumulative sum of relevant items up to the i-th position
    # np.arange create an array of the specified length with a[i]=i
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])
    return map_score

# Average Precision
def AP(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
    ap_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])
    return ap_score

# <--------------------------------------------------------------------------------------------------> #
## RECOMMENDERS
URM_train = [[]]

# Random Recommender System:
class RandomRecommender(object):
    def fit(self, URM_train):
        self.n_items = URM_train.shape[1]
    
    def recommend(self, user_id, at=5): 
        recommended_items = np.random.choice(self.n_items, at)
        return recommended_items

randomRecommender = RandomRecommender()
randomRecommender.fit(URM_train)

# Evaluation function to test it
def evaluate_algorithm(URM_test, recommender_object, at=5):
    
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_AP = 0.0
    
    num_eval = 0


    for user_id in range(URM_test.shape[0]):

        # relevant_items = indices of the items. np.indptr say where to find the column for user_id.
        #CSR matrix (Compressed Sparse Row matrix): data, indices, indptr
        relevant_items = URM_test.indices[URM_test.indptr[user_id]:URM_test.indptr[user_id+1]]
        
        if len(relevant_items)>0:
            
            recommended_items = recommender_object.recommend(user_id, at=at)
            num_eval+=1

            cumulative_precision += precision(recommended_items, relevant_items)
            cumulative_recall += recall(recommended_items, relevant_items)
            cumulative_AP += AP(recommended_items, relevant_items)
            
    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    MAP = cumulative_AP / num_eval
    
    print("Recommender results are: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
        cumulative_precision, cumulative_recall, MAP)) 


# Top Popular Recommender System:
class TopPopRecommender(object):

    def fit(self, URM_train):
        # CSC Compressed Sparse Column matrix
        # ediff1d function is useful for finding the differences between successive elements of an array
        # URM_train.tocsc().indptr will contain the indeces of where an item ID start and finish
        item_popularity = np.ediff1d(URM_train.tocsc().indptr)

        # self.popular_items = np.argsort(item_popularity)
        self.popular_items = np.argsort(item_popularity)
        self.popular_items = np.flip(self.popular_items, axis = 0)
    

    def recommend(self, user_id, at=5):
        recommended_items = self.popular_items[0:at]
        return recommended_items



# Improving the first version of Top Popular technique by removing items already "seen" :
class TopPopRecommender(object):

    def fit(self, URM_train):
        
        self.URM_train = URM_train
        item_popularity = np.ediff1d(URM_train.tocsc().indptr)
        self.popular_items = np.argsort(item_popularity)
        self.popular_items = np.flip(self.popular_items, axis = 0)
    
    
    def recommend(self, user_id, at=5, remove_seen=True):
        if remove_seen:
            seen_items = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id+1]]
            unseen_items_mask = np.in1d(self.popular_items, seen_items,
                                        assume_unique=True, invert = True)
            unseen_items = self.popular_items[unseen_items_mask]
            recommended_items = unseen_items[0:at]

        else:
            recommended_items = self.popular_items[0:at]

        return recommended_items



# Global Effects Recommender System:
# Removing Global average from each rating:
globalAverage = np.mean(URM_train.data)
URM_train_unbiased = URM_train.copy()
URM_train_unbiased.data -= globalAverage
#Item Bias:
# This computes the mean of the column including zero values
item_mean_rating = URM_train_unbiased.mean(axis=0)
# User bias:
# This computes the mean of the row including zero values
user_mean_rating = URM_train_unbiased.mean(axis=1)

class GlobalEffectsRecommender(object):
    def fit(self, URM_train):
        self.URM_train = URM_train

        globalAverage = np.mean(URM_train.data)
        URM_train_unbiased = URM_train.copy()
        URM_train_unbiased.data -= globalAverage 
        
        # User Bias
        user_mean_rating = URM_train_unbiased.mean(axis=1)
        user_mean_rating = np.array(user_mean_rating).squeeze()
        # In order to apply the user bias we have to change the rating value 
        # in the URM_train_unbiased inner data structures
        # If we were to write:
        # URM_train_unbiased[user_id].data -= user_mean_rating[user_id]
        # we would change the value of a new matrix with no effect on the original data structure
        for user_id in range(len(user_mean_rating)):
            start_position = URM_train_unbiased.indptr[user_id]
            end_position = URM_train_unbiased.indptr[user_id+1]
            URM_train_unbiased.data[start_position:end_position] -= user_mean_rating[user_id]

        #Item Bias
        item_mean_rating = URM_train_unbiased.mean(axis=0)
        item_mean_rating = np.array(item_mean_rating).squeeze()

        self.bestRatedItems = np.argsort(item_mean_rating)
        self.bestRatedItems = np.flip(self.bestRatedItems, axis = 0)

        
    def recommend(self, user_id, at=5, remove_seen=True):
        if remove_seen:
            seen_items = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id+1]]
            unseen_items_mask = np.in1d(self.bestRatedItems, seen_items,
                                        assume_unique=True, invert = True)
            unseen_items = self.bestRatedItems[unseen_items_mask]
            recommended_items = unseen_items[0:at]
        else:
            recommended_items = self.bestRatedItems[0:at]

        return recommended_items


# ----------------------------------------------- #
#                    CBF                          #
# ----------------------------------------------- #
# Content Based Filtering Recommender System: CBF. It uses ICM to create item-item similarity matrix
ICM_all = [[]]

# KNN build with the following steps:
item_id = 80
shrink = 10

# dot product. ravel() to change a 2-dimensional array or a multi-dimensional array into a contiguous flattened array
numerator_vector = ICM_all[item_id].dot(ICM_all.T).toarray().ravel()	 
#norm
item_norms = np.sqrt(np.array(ICM_all.T.power(2).sum(axis=0))).ravel()
# product of the norms + shrink term + term to make denominator different from zero
denominator_vector = item_norms[item_id] * item_norms + shrink + 1e-6

similarity_vector = numerator_vector/denominator_vector
sorted_item_indices = np.argsort(-similarity_vector)


# ----------------------------------------------- #
#                    CF                          #
# ----------------------------------------------- #
# Collaborative Filtering (CF): Item Based
# Main source: URM. Similarity: i*j/ ( ||i||_2*||j||_2 )
URM = [[]]
item_id = 80
shrink = 10

numerator_vector = URM_train.T[item_id].dot(URM_train).toarray().ravel()
item_norms = np.sqrt(np.array(URM_train.power(2).sum(axis=0))).ravel()
denominator_vector = item_norms[item_id] * item_norms + shrink + 1e-6

similarity_vector = numerator_vector/denominator_vector
sorted_item_indices = np.argsort(-similarity_vector)

# Collaborative Filtering (CF): User Based
# Main source: URM, Similarity: u*v/ ( ||u||_2*||v||_2 )

user_id = 80
shrink = 10

numerator_vector = URM_train[user_id].dot(URM_train.T).toarray().ravel()
item_norms = np.sqrt(np.array(URM_train.T.power(2).sum(axis=0))).ravel()
denominator_vector = item_norms[item_id] * item_norms + shrink + 1e-6

# ----------------------------------------------- #
#                    SLIM MSE                     #
# ----------------------------------------------- #
# Item Based Collaborative Filtering using ML
# SLIM: Sparse LInear Method. Loss function is MSE: E(S)=|| R - R*S ||_2

# First load the usual data, then compute the item-item siilarity matrix, compute prediction, update rule and finally training loop. 
n_users, n_items = URM_train.shape
# create a dense similarity matrix, initialized as zero
item_item_S = np.zeros((n_items, n_items), dtype = np.float)
# Sample an interaction and compute the prediction of the current SLIM model
URM_train_coo = URM_train.tocoo() #COOrdinate format: row, col, data
learning_rate = 1e-6
loss = 0.0
start_time = time.time()
for sample_num in range(100000):
    
    # Randomly pick sample
    sample_index = np.random.randint(URM_train_coo.nnz)
    user_id = URM_train_coo.row[sample_index]
    item_id = URM_train_coo.col[sample_index]
    true_rating = URM_train_coo.data[sample_index]

    # Compute prediction
    predicted_rating = URM_train[user_id].dot(item_item_S[:,item_id])[0]
        
    # Compute prediction error, or gradient
    prediction_error = true_rating - predicted_rating
    loss += prediction_error**2
    
    # Update model, in this case the similarity
    items_in_user_profile = URM_train[user_id].indices
    ratings_in_user_profile = URM_train[user_id].data
    # Gradient Descent formula: x = x + alpha * epsilon * x
    item_item_S[items_in_user_profile,item_id] += learning_rate * prediction_error * ratings_in_user_profile

    #To improve the training time
    #items_in_user_profile = URM_train.indices[URM_train.indptr[user_id]:URM_train.indptr[user_id+1]]
    #ratings_in_user_profile = URM_train.data[URM_train.indptr[user_id]:URM_train.indptr[user_id+1]]
    #item_item_S[items_in_user_profile,item_id] += learning_rate * prediction_error * ratings_in_user_profile

# ----------------------------------------------- #
#                        MF                       #
# ----------------------------------------------- #
# Matrix Factorization: R=X*Y (or WH), where X is the matrix of users latent factors or features and Y of items features
# How to choose X and Y? Minimize loss function (e.g. MSE)

# ----------------------------------------------- #
# MF WITH FUNKSVD (Funk Singular Value Decomposition) #
# Loss function: |R-WH|_F + alpha |W|_2 + beta |H|_2

URM_train_coo = URM_train.tocoo()
num_factors = 10
learning_rate = 1e-4
# RANDOM NON-ZERO VALUES otherwise all updates will be zero and the model will not be able to learn
user_factors = np.random.random((n_users, num_factors))
item_factors = np.random.random((n_items, num_factors))
loss = 0.0
regularization = 1e-5
start_time = time.time()

for sample_num in range(1000000):
    # Randomly pick sample
    sample_index = np.random.randint(URM_train_coo.nnz)
    user_id = URM_train_coo.row[sample_index]
    item_id = URM_train_coo.col[sample_index]
    rating = URM_train_coo.data[sample_index]
    # Compute prediction
    predicted_rating = np.dot(user_factors[user_id,:], item_factors[item_id,:])
    # Compute prediction error, or gradient
    prediction_error = rating - predicted_rating
    loss += prediction_error**2
    # Copy original value to avoid messing up the updates
    H_i = item_factors[item_id,:]
    W_u = user_factors[user_id,:]  
    
    # Update the matrices using gradient descent
    user_factors_update = prediction_error * H_i - regularization * W_u
    item_factors_update = prediction_error * W_u - regularization * H_i
    user_factors[user_id,:] += learning_rate * user_factors_update 
    item_factors[item_id,:] += learning_rate * item_factors_update    



# ----------------------------------------------- #
# MF WITH iALS (implicit alternating least squares)
# Each update is done for a user, then for an item
# Loss function: L = 1/2 * ||C*(R-XY)||_F + reg*(||X||_2 + ||Y||_2 )
n_users, n_items = URM_train.shape
num_factors = 10
user_factors = np.random.random((n_users, num_factors))
item_factors = np.random.random((n_items, num_factors))

# We need to define a confidence matrix, e.g. linear or popularity based
def linear_confidence_function(URM_train, alpha):
    URM_train.data = 1.0 + alpha*URM_train.data
    return URM_train
C_URM_train = linear_confidence_function(URM_train, 0.5)
C_URM_train_csc = C_URM_train.tocsc()

# Function to update a row: (YtY + Yt*(Cu-I)*Y + reg*I)^-1 * Yt*Cu
def _update_row(interaction_profile, interaction_confidence, Y, YtY, regularization_diagonal):
    Y_interactions = Y[interaction_profile, :]
    A = Y_interactions.T.dot(((interaction_confidence - 1) * Y_interactions.T).T)
    B = YtY + A + regularization_diagonal
    return np.dot(np.linalg.inv(B), Y_interactions.T.dot(interaction_confidence))

# Regularization diagonal: we need the regularization coefficient only on the diagonal
regularization_coefficient = 1e-4
regularization_diagonal = np.diag(regularization_coefficient * np.ones(num_factors))

# VV and UU
VV = item_factors.T.dot(item_factors)
UU = user_factors.T.dot(user_factors)

for n_epoch in range(10):
    start_time = time.time()
    for user_id in range(C_URM_train.shape[0]):
        start_pos = C_URM_train.indptr[user_id]
        end_pos = C_URM_train.indptr[user_id + 1]
        user_profile = C_URM_train.indices[start_pos:end_pos]
        user_confidence = C_URM_train.data[start_pos:end_pos]
        user_factors[user_id, :] = _update_row(user_profile, user_confidence, item_factors, VV, regularization_diagonal)

    for item_id in range(C_URM_train.shape[1]):
        start_pos = C_URM_train_csc.indptr[item_id]
        end_pos = C_URM_train_csc.indptr[item_id + 1]
        item_profile = C_URM_train_csc.indices[start_pos:end_pos]
        item_confidence = C_URM_train_csc.data[start_pos:end_pos]
        item_factors[item_id, :] = _update_row(item_profile, item_confidence, user_factors, UU, regularization_diagonal)    


# ----------------------------------------------- #
# Pure SVD using scikit randomized_svd: R=U*Sigma*VT
# Step one and only: Compute the decomposition

n_users, n_items = URM_train.shape
from sklearn.utils.extmath import randomized_svd
# This algorithm finds a (usually very good) approximate truncated singular value 
# decomposition using randomization to speed up the computations. It is particularly fast 
# on large matrices on which you wish to extract only a small number of components
num_factors = 10
U, Sigma, VT = randomized_svd(URM_train, n_components=num_factors)
user_factors = np.dot(U, np.diag(Sigma))
item_factors = VT
predicted_rating_mf = np.dot(user_factors[user_id,:], item_factors[:,item_id])
# Item-based similarity with PureSVD
item_item_similarity = np.dot(VT.T,VT)
predicted_rating_similarity = URM_train[user_id,:].dot(item_item_similarity[:,item_id])

# ----------------------------------------------- #
#                     BPR                         #
# ----------------------------------------------- #
# BPR: Bayesian Probabilistic Ranking
# Objective: maximize probability of retrivieng a relevant item i for user u instead of non-relevant item j
# Loss function: sigmoid of predicted rating for i - predicted rating of j (pairwise difference)
# For implicit ratings: i=observed, j=not-observed
# Sthocastic gradient descent because otherwise lot of triplets with popular items
# Basically we do: 
    # - Increase the relevance (according to $u$) of features belonging to $i$ but not to $j$ and vice-versa
    # - Increase the relevance of features assigned to $i$
    # - Decrease the relevance of features assigned to $j$

# SLIM BPR (Sparse LInear Method with Bayesian Probabilistic Ranking)
# SLIM build an item-item similarity matrix
# Predicted xuij = Ruk*Ski - Ruk*Skj

n_users, n_items = URM_train.shape
item_item_S = np.zeros((n_items, n_items), dtype = np.float) # starting item-item S of zeros
# Mask of positive interactions (it depends on data)
URM_mask = URM_train.copy()
URM_mask.data[URM_mask.data <= 3] = 0
URM_mask.eliminate_zeros()
user_id = np.random.choice(n_users)
user_seen_items = URM_mask.indices[URM_mask.indptr[user_id]:URM_mask.indptr[user_id+1]]
pos_item_id = np.random.choice(user_seen_items)
neg_item_selected = False
# It's faster to just try again then to build a mapping of the non-seen items
while (not neg_item_selected):
    neg_item_id = np.random.randint(0, n_items)
    if (neg_item_id not in user_seen_items):
        neg_item_selected = True
x_ui = item_item_S[pos_item_id, user_seen_items].sum()
x_uj = item_item_S[neg_item_id, user_seen_items].sum()
x_uij = x_ui - x_uj
sigmoid_item = 1 / (1 + np.exp(x_uij))
learning_rate = 1e-3
# Update positive items
item_item_S[pos_item_id, user_seen_items] += learning_rate * sigmoid_item
item_item_S[pos_item_id, pos_item_id] = 0 # Remember to set the diagonal to 0
# Update negative items
item_item_S[neg_item_id, user_seen_items] -= learning_rate * sigmoid_item
item_item_S[neg_item_id, neg_item_id] = 0 # Remember to set the diagonal to 0
# we should iterate this



# BPR FOR MF (Bayesian Probabilistic Ranking for Matrix Factorization)
# We need two matrices, for user latent factors and for item latent factors
num_factors = 10
user_factors = np.random.random((n_users, num_factors))
item_factors = np.random.random((n_items, num_factors))
URM_mask = URM_train.copy()
URM_mask.data[URM_mask.data <= 3] = 0
URM_mask.eliminate_zeros()
user_id = np.random.choice(n_users)
user_seen_items = URM_mask.indices[URM_mask.indptr[user_id]:URM_mask.indptr[user_id+1]]
pos_item_id = np.random.choice(user_seen_items)
neg_item_selected = False
x_ui = np.dot(user_factors[user_id,:], item_factors[pos_item_id,:])
x_uj = np.dot(user_factors[user_id,:], item_factors[neg_item_id,:])
x_uij = x_ui - x_uj
sigmoid_item = 1 / (1 + np.exp(x_uij))
regularization = 1e-4
learning_rate = 1e-2
H_i = item_factors[pos_item_id,:]
H_j = item_factors[neg_item_id,:]
W_u = user_factors[user_id,:]
user_factors[user_id,:] += learning_rate * (sigmoid_item * ( H_i - H_j ) - regularization * W_u)
item_factors[pos_item_id,:] += learning_rate * (sigmoid_item * ( W_u ) - regularization * H_i)
item_factors[neg_item_id,:] += learning_rate * (sigmoid_item * (-W_u ) - regularization * H_j)



# MF with Pytorch
# MF models rely upon latent factors for users and items which are called 'embeddings'

num_factors = 10
n_users, n_items = URM_train.shape
import torch
# Creates U
user_factors = torch.nn.Embedding(num_embeddings=n_users, embedding_dim=num_factors)
# Creates V
item_factors = torch.nn.Embedding(num_embeddings=n_items, embedding_dim=num_factors)
# To compute the prediction we have to multiply the user factors to the item factors, 
# which is a linear operation. We define a single layer and an activation function, 
# which takes the result and transforms it in the final prediction. 
# The activation function can be used to restrict the predicted values
layer_1 = torch.nn.Linear(in_features=num_factors, out_features=1)
activation_function = torch.nn.ReLU()
# 1. Define a list of user/item indices.
item_index = [15]
user_index = [42]
# 2. Create a tensor from it. Specify indices are of type int64.
user_index = torch.Tensor(user_index).type(torch.int64)
item_index = torch.Tensor(item_index).type(torch.int64)
# 3. Get the user and item embeddings 
current_user_factors = user_factors(user_index)
current_item_factors = item_factors(item_index)
# 4. Compute the element-wise product of the embeddings
element_wise_product = torch.mul(current_user_factors, current_item_factors)
# 5. Pass the element-wise product of the embeddings
prediction = layer_1(element_wise_product)
# 6. Pass the output of the single layer network to the activation function
prediction = activation_function(prediction)
# To take the result of the prediction and transform it into a traditional numpy array 
# you have to first call .detach() and then .numpy()
prediction_numpy = prediction.detach().numpy()



# ----------------------------------------------- #
#                   HYBRIDS                       #
# ----------------------------------------------- #

# HYBRID with STACKING 

# A simple idea is to look at both iteractions and features as a similar type of information. 
# Given an item, you can see the users that interacted with it as features of that item, 
# in the same way as its genre.
# To do so, you can concatenate URM and ICM in a single data structure of shape 
# WARNING: You are creating a new matrix that can be used as URM or ICM interchangeably
# and plugged into any model you already have. This gives you flexibility but requires caution

# If one of the two is much larger than the other (like 60k users vs 20 features) the smaller 
# will be overwhelmed numerically. Add a weight to the smaller one to increase its importance 
# (will only work for models that allow explicit interactions, will not work with models that use implicit ones)
import scipy.sparse as sps
ICM_genres = [[]]
stacked_URM = sps.vstack([URM_train, ICM_genres.T])
stacked_URM = sps.csr_matrix(stacked_URM)
stacked_ICM = sps.csr_matrix(stacked_URM.T)
# Collaborative Filtering: stacked_URM
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
recommender_ItemKNNCF = ItemKNNCFRecommender(stacked_URM)
recommender_ItemKNNCF.fit()
result_df, _ = evaluator_validation.evaluateRecommender(recommender_ItemKNNCF)
# Content Based Filtering: URM_train + stacked_ICM
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
recommender_ItemKNNCBF = ItemKNNCBFRecommender(URM_train, stacked_ICM)
recommender_ItemKNNCBF.fit()
result_df, _ = evaluator_validation.evaluateRecommender(recommender_ItemKNNCBF)



# HYBRID OF MODELS WITH THE SAME STRUCTURE
# For instance, merge item-item similarity of Item CF and P3Alpha
itemKNNCF = ItemKNNCFRecommender(URM_train)
itemKNNCF.fit()
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
P3alpha = P3alphaRecommender(URM_train)
P3alpha.fit()
alpha = 0.7
new_similarity = (1 - alpha) * itemKNNCF.W_sparse + alpha * P3alpha.W_sparse
from Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
recommender_object = ItemKNNCustomSimilarityRecommender(URM_train)
recommender_object.fit(new_similarity)


# HYBRID WITH DIFFERENT STRUCTURE: use the item score
# There are certain cases where we want to combine recommenders with different underlying structures
# For instance ItemKNN and UserKNN or ItemKNN with PureSVD.
# In this case we cannot combine the model parameters themselves, rather we can combine the predictions.
# We can combine the predictions via a weighted average
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
pureSVD = PureSVDRecommender(URM_train)
pureSVD.fit()
user_id = 50
item_scores_itemknn = itemKNNCF._compute_item_score(user_id)
item_scores_puresvd = pureSVD._compute_item_score(user_id)
alpha = 0.7
new_item_scores = alpha * item_scores_itemknn + (1 - alpha) * item_scores_puresvd


# MODELS WITH RATING PREDICTIONS VS RANKING LOSS FUNCTION
# Sometimes we have models that optimize different loss functions, one BPR (ranking) and another rating prediction (MSE). For instance:
# - SLIMBPR and SLIM EN/MSE
# - FunkSVD and MFBPR
# This brings another important problem, for rank-based models there is no clear meaning for the *absolute value* of the prediction. 
# Need normalization on the weights, to ensure that the values are, at least, in the same range for all models. 

# Try to merge FunkSVD with SLIM_BPR
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_FunkSVD_Cython
funk_svd_recommender = MatrixFactorization_FunkSVD_Cython(URM_train)
funk_svd_recommender.fit(epochs=200)
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
slim_bpr_recommender = SLIM_BPR_Cython(URM_train)
slim_bpr_recommender.fit(epochs=200)
# item_score for FunkSVD
funk_svd_score = funk_svd_recommender._compute_item_score(user_id)
funk_svd_score
# item_score for SLIM_BPR
slim_bpr_score = slim_bpr_recommender._compute_item_score(user_id).flatten()
slim_bpr_score
# Normalize scores for both models
l1_funk_svd = LA.norm([funk_svd_score], 1)
l1_funk_svd_scores = funk_svd_score / l1_funk_svd
l1_slim_bpr = LA.norm(slim_bpr_score, 1)
l1_slim_bpr_scores = slim_bpr_score / l1_slim_bpr

lambda_weights = 0.66
l1_new_scores = lambda_weights * l1_slim_bpr_scores + (1 - lambda_weights) * l1_funk_svd_scores


# USER-WISE HYBRIDS
# Models do not have the same accuracy for different user types. 
# Let's divide the users according to their profile length and then compare the recommendation quality we get from a CF model.
# Then use different models for different groups.