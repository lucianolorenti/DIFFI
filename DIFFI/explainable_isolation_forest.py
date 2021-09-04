import numpy as np


from sklearn.ensemble import IsolationForest 
from DIFFI.utils import compute_node_depth

class DIFFI:
    def __init__(self, model:IsolationForest):
        self.model = model

    def explain(self, X):
        """
        Returns
        --------
        diffi_te:
        
        ord_idx_diffi_te:
        """
        y_pred = np.array(self.model.decision_function(X) < 0).astype('int')        
        return self._local_diffi(X[np.where(y_pred == 1)])

    def _local_diffi(self, X):
        """
        Compute feature importance 

        Parameters
        ----------
        
        """
        fi = []
        ord_idx = []
        for i in range(X.shape[0]):
            x_curr = X[i, :]
            fi_curr = self.local_diffi_one_sample(x_curr)
            fi.append(fi_curr)
            ord_idx_curr = np.argsort(fi_curr)[::-1]
            ord_idx.append(ord_idx_curr)
        fi = np.vstack(fi)
        ord_idx = np.vstack(ord_idx)
        return fi, ord_idx

    def local_diffi_one_sample(self, x):
        """
        Compute fi for one sample

        Parameters
        ----------
        x: np.array Sample of data to compute the feature importance

        """
        # initialization 
        cfi = np.zeros(len(x), dtype='float')
        counter = np.zeros(len(x), dtype='int')
        max_depth = int(np.ceil(np.log2(self.model.max_samples)))
        for estimator in self.model.estimators_:
            feature = estimator.tree_.feature
            node_depth, is_leaves= compute_node_depth(estimator.tree_)
            
            x = x.reshape(1,-1)
            node_indicator = estimator.decision_path(x)
            node_indicator_array = node_indicator.toarray()
            path = list(np.where(node_indicator_array == 1)[1])
            leaf_depth = node_depth[path[-1]]
            for node in path:
                if is_leaves[node]:
                    continue
                current_feature = feature[node] 
                cfi[current_feature] += (1 / leaf_depth) - (1 / max_depth)
                counter[current_feature] += 1

        fi = [cfi[i] / counter[i] if counter[i] != 0 else 0 
            for i in range(len(cfi)) ]
        return np.array(fi)

    
    

  