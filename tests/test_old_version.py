import os
import time
from math import ceil

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score

from DIFFI.data import get_fs_dataset
from DIFFI.global_score import diffi_inbag
from .sklearn_mod_functions import decision_function_single_tree


def _get_iic(estimator, predictions, is_leaves, adjust_iic):
    desired_min = 0.5
    desired_max = 1.0
    epsilon = 0.0
    n_nodes = estimator.tree_.node_count
    lambda_ = np.zeros(n_nodes)
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    # compute samples in each node
    node_indicator_all_samples = estimator.decision_path(predictions).toarray()
    num_samples_in_node = np.sum(node_indicator_all_samples, axis=0)
    # ASSIGN INDUCED IMBALANCE COEFFICIENTS (IIC)
    for node in range(n_nodes):
        # compute relevant quantities for current node
        num_samples_in_current_node = num_samples_in_node[node]
        num_samples_in_left_children = num_samples_in_node[children_left[node]]
        num_samples_in_right_children = num_samples_in_node[children_right[node]]
        # if there is only 1 feasible split or node is leaf -> no IIC is assigned
        if (
            num_samples_in_current_node == 0
            or num_samples_in_current_node == 1
            or is_leaves[node]
        ):
            lambda_[node] = -1
        # if useless split -> assign epsilon
        elif num_samples_in_left_children == 0 or num_samples_in_right_children == 0:
            lambda_[node] = epsilon
        else:
            if num_samples_in_current_node % 2 == 0:  # even
                current_min = 0.5
            else:  # odd
                current_min = (
                    ceil(num_samples_in_current_node / 2) / num_samples_in_current_node
                )
            current_max = (
                num_samples_in_current_node - 1
            ) / num_samples_in_current_node
            tmp = (
                np.max([num_samples_in_left_children, num_samples_in_right_children])
                / num_samples_in_current_node
            )
            if adjust_iic and current_min != current_max:
                lambda_[node] = ((tmp - current_min) / (current_max - current_min)) * (
                    desired_max - desired_min
                ) + desired_min
            else:
                lambda_[node] = tmp
    return lambda_


def diffi_ib(iforest, X, adjust_iic=True):  # "ib" stands for "in-bag"
    # start time
    start = time.time()
    # initialization
    num_feat = X.shape[1]
    estimators = iforest.estimators_
    cfi_outliers_ib = np.zeros(num_feat).astype("float")
    cfi_inliers_ib = np.zeros(num_feat).astype("float")
    counter_outliers_ib = np.zeros(num_feat).astype("int")
    counter_inliers_ib = np.zeros(num_feat).astype("int")
    in_bag_samples = iforest.estimators_samples_
    # for every iTree in the iForest
    for k, estimator in enumerate(estimators):
        # get in-bag samples indices
        in_bag_sample = list(in_bag_samples[k])
        # get in-bag samples (predicted inliers and predicted outliers)
        X_ib = X[in_bag_sample, :]
        as_ib = decision_function_single_tree(iforest, k, X_ib)
        X_outliers_ib = X_ib[np.where(as_ib < 0)]
        X_inliers_ib = X_ib[np.where(as_ib > 0)]
        if X_inliers_ib.shape[0] == 0 or X_outliers_ib.shape[0] == 0:
            continue
        # compute relevant quantities
        n_nodes = estimator.tree_.node_count
        children_left = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature = estimator.tree_.feature
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        # compute node depths
        stack = [(0, -1)]
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1
            # if we have a test node
            if children_left[node_id] != children_right[node_id]:
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                is_leaves[node_id] = True
        # OUTLIERS
        # compute IICs for outliers
        lambda_outliers_ib = _get_iic(estimator, X_outliers_ib, is_leaves, adjust_iic)
        # update cfi and counter for outliers
        node_indicator_all_points_outliers_ib = estimator.decision_path(X_outliers_ib)
        node_indicator_all_points_array_outliers_ib = (
            node_indicator_all_points_outliers_ib.toarray()
        )
        # for every point judged as abnormal
        for i in range(len(X_outliers_ib)):
            path = list(
                np.where(node_indicator_all_points_array_outliers_ib[i] == 1)[0]
            )
            depth = node_depth[path[-1]]
            for node in path:
                current_feature = feature[node]
                if lambda_outliers_ib[node] == -1:
                    continue
                else:
                    cfi_outliers_ib[current_feature] += (
                        1 / depth
                    ) * lambda_outliers_ib[node]
                    counter_outliers_ib[current_feature] += 1
        # INLIERS
        # compute IICs for inliers
        lambda_inliers_ib = _get_iic(estimator, X_inliers_ib, is_leaves, adjust_iic)
        # update cfi and counter for inliers
        node_indicator_all_points_inliers_ib = estimator.decision_path(X_inliers_ib)
        node_indicator_all_points_array_inliers_ib = (
            node_indicator_all_points_inliers_ib.toarray()
        )
        # for every point judged as normal
        for i in range(len(X_inliers_ib)):
            path = list(np.where(node_indicator_all_points_array_inliers_ib[i] == 1)[0])
            depth = node_depth[path[-1]]
            for node in path:
                current_feature = feature[node]
                if lambda_inliers_ib[node] == -1:
                    continue
                else:
                    cfi_inliers_ib[current_feature] += (1 / depth) * lambda_inliers_ib[
                        node
                    ]
                    counter_inliers_ib[current_feature] += 1
    # compute FI
    fi_outliers_ib = np.where(
        counter_outliers_ib > 0, cfi_outliers_ib / counter_outliers_ib, 0
    )
    fi_inliers_ib = np.where(
        counter_inliers_ib > 0, cfi_inliers_ib / counter_inliers_ib, 0
    )
    fi_ib = fi_outliers_ib / fi_inliers_ib
    end = time.time()
    exec_time = end - start
    return fi_ib, exec_time


def logarithmic_scores(fi):
    # fi is a (N x p) matrix, where N is the number of runs and p is the number of features
    num_feats = fi.shape[1]
    p = np.arange(1, num_feats + 1, 1)
    log_s = [1 - (np.log(x)/np.log(num_feats)) for x in p]
    scores = np.zeros(num_feats)
    for i in range(fi.shape[0]):
        sorted_idx = np.flip(np.argsort(fi[i,:]))
        for j in range(num_feats):
            curr_feat = sorted_idx[j]
            if fi[i,curr_feat]>0:
                scores[curr_feat] += log_s[j]
    return scores 




def diffi_ranks(X, y, n_trees, max_samples, n_iter):
    def compute_ranking(d):
        d = np.vstack(d)
        scores = logarithmic_scores(d)
        sorted_idx = np.flip(np.argsort(scores))
        return scores, sorted_idx

    f1_all, fi_diffi_all = [], []
    fi_diffi_new_all = []
    for k in range(n_iter):
        # ISOLATION FOREST
        # fit the model
        iforest = IsolationForest(n_estimators= n_trees, max_samples=max_samples, 
                                  contamination='auto', random_state=k)
        iforest.fit(X)
        # get predictions
        y_pred = np.array(iforest.decision_function(X) < 0).astype('int')
        # get performance metrics
        f1_all.append(f1_score(y, y_pred))
        # diffi
        fi_diffi, _ = diffi_ib(iforest, X, adjust_iic=True)
        new_fi_diffi = diffi_inbag(iforest, X, adjust_iic=True)
        fi_diffi_all.append(fi_diffi)
        fi_diffi_new_all.append(new_fi_diffi)
    
    avg_f1 = np.mean(f1_all)
    scores, sorted_idx = compute_ranking(fi_diffi_all)
    scores_new, sorted_idx_new = compute_ranking(fi_diffi_new_all)


    return sorted_idx, sorted_idx_new, avg_f1


def test_diffi_ib():
    dataset_id = 'lympho'

    X, y, contamination = get_fs_dataset(dataset_id, seed=42)
    sorted_idx, sorted_idx_new, avg_f1_ranking = diffi_ranks(X, y, n_trees=100, max_samples=256, n_iter=5)
    assert( (sorted_idx==sorted_idx_new).all())
