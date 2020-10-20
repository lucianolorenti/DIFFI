import numpy as np
from sklearn.metrics import f1_score
from sklearn.base import OutlierMixin
from sklearn.ensemble import IsolationForest 
from DIFFI.utils import diffi_inbag, logarithmic_scores, local_diffi

class ExplainableIsolationForest(IsolationForest):
    def explain(self, X):
        y_pred = np.array(self.decision_function(X) < 0).astype('int')        
        return local_diffi(self, X[np.where(y_pred == 1)])

    
    

  