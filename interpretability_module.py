import numpy as np
import time
from math import ceil
from utils import *
from sklearn_mod_functions import *


def compute_node_depth(itree):
    """
    Compute the nodes depth and the leafs of a given isolation tree

    Parameters
    ----------
    itree: 
    """
    n_nodes = itree.node_count
    children_left = itree.children_left
    children_right = itree.children_right
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1
        # if we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True
    return (node_depth, is_leaves)





def diffi_ib(iforest, X, adjust_iic=True):  # "ib" stands for "in-bag"
    # start time
    start = time.time()
    # initialization
    num_feat = X.shape[1]
    cfi_outliers_ib = np.zeros(num_feat, dtype=np.float)
    cfi_inliers_ib = np.zeros(num_feat, dtype=np.float)
    counter_outliers_ib = np.zeros(num_feat, dtype=np.int)
    counter_inliers_ib = np.zeros(num_feat, dtype=np.int)

    # for every iTree in the iForest
    for k, estimator in enumerate(iforest.estimators_):
        # get in-bag samples indices
        in_bag_sample = list(iforest.estimators_samples_[k])
        # get in-bag samples (predicted inliers and predicted outliers)
        X_ib = X[in_bag_sample, :]
        as_ib = decision_function_single_tree(iforest, k, X_ib)
        X_outliers_ib = X_ib[np.where(as_ib < 0)]
        X_inliers_ib = X_ib[np.where(as_ib > 0)]
        if X_inliers_ib.shape[0] == 0 or X_outliers_ib.shape[0] == 0:
            continue
        # compute relevant quantities
        node_depth, is_leaves = compute_node_depth(estimator.tree_)

        update_outiers_counters()

        update_inliers_counters()

    # compute FI
    fi_outliers_ib = np.where(counter_outliers_ib > 0,
                              cfi_outliers_ib / counter_outliers_ib, 0)
    fi_inliers_ib = np.where(counter_inliers_ib > 0,
                             cfi_inliers_ib / counter_inliers_ib, 0)
    fi_ib = fi_outliers_ib / fi_inliers_ib
    end = time.time()
    exec_time = end - start
    return fi_ib, exec_time


def local_diffi(iforest : IsolationForest, x):


    estimators = iforest.estimators_
    cfi = np.zeros(len(x)).astype('float')
    counter = np.zeros(len(x)).astype('int')
    max_depth = int(np.ceil(np.log2(iforest.max_samples)))
    # for every iTree in the iForest
    for estimator in estimators:
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
            # if test node
            if (children_left[node_id] != children_right[node_id]):
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                is_leaves[node_id] = True
        # update cumulative importance and counter
        x = x.reshape(1, -1)
        node_indicator = estimator.decision_path(x)
        node_indicator_array = node_indicator.toarray()
        path = list(np.where(node_indicator_array == 1)[1])
        leaf_depth = node_depth[path[-1]]
        for node in path:
            if not is_leaves[node]:
                current_feature = feature[node]
                cfi[current_feature] += (1 / leaf_depth) - (1 / max_depth)
                counter[current_feature] += 1
    # compute FI
    fi = np.zeros(len(cfi))
    for i in range(len(cfi)):
        if counter[i] != 0:
            fi[i] = cfi[i] / counter[i]
    end = time.time()
    return fi


def _get_iic(estimator,
             predictions,
             is_leaves,
             adjust_iic,
             epsilon=0,
             desired_min=0.5,
             desired_max=1.0):
    """
    Comptue the IIC

    Arguments
    ---------
    estimator:
    """
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
        num_samples_in_right_children = num_samples_in_node[
            children_right[node]]
        # if there is only 1 feasible split or node is leaf -> no IIC is assigned
        if num_samples_in_current_node == 0 or num_samples_in_current_node == 1 or is_leaves[
                node]:
            lambda_[node] = -1
        # if useless split -> assign epsilon
        elif num_samples_in_left_children == 0 or num_samples_in_right_children == 0:
            lambda_[node] = epsilon
        else:
            if num_samples_in_current_node % 2 == 0:  # even
                current_min = 0.5
            else:  # odd
                current_min = ceil(num_samples_in_current_node /
                                   2) / num_samples_in_current_node
            current_max = (num_samples_in_current_node -
                           1) / num_samples_in_current_node
            tmp = np.max([
                num_samples_in_left_children, num_samples_in_right_children
            ]) / num_samples_in_current_node
            if adjust_iic and current_min != current_max:
                lambda_[node] = ((tmp - current_min) /
                                 (current_max - current_min)) * (
                                     desired_max - desired_min) + desired_min
            else:
                lambda_[node] = tmp
    return lambda_