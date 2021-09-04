from DIFFI import PACKAGE_PATH
from scipy.io import loadmat
from sklearn.utils import shuffle
import logging
import pickle 
import numpy as np 

logger = logging.getLogger(__name__)

VALID_DATASETS = set(["cardio", "ionosphere", "letter", "musk", "lympho", "satellite"])

def load_syntethic_dataset():
    with open(PACKAGE_PATH / "datasets" / "local" / 'syn_train.pkl', 'rb') as f:
        data_tr = pickle.load(f)
    with open(PACKAGE_PATH / "datasets" / "local" / 'syn_test.pkl', 'rb') as f:
        data_te = pickle.load(f)

    X_tr = data_tr['X']
    y_tr = data_tr['y']
    # test outliers
    X_xaxis = data_te['X_xaxis']
    X_yaxis = data_te['X_yaxis']
    X_bisec = data_te['X_bisec']
    return X_tr, y_tr, X_xaxis, X_yaxis, X_bisec

def load_glass_dataset():
    """We consider data points in the "headlamps glass" class (class 7 in the original dataset) as test outliers. 
    

    Returns
    -------
    [type]
        [description]
    """
    with open(PACKAGE_PATH / "datasets" / "local" / 'glass.pkl', 'rb') as f:
        data = pickle.load(f)
    # training data (inliers and outliers)
    X_tr = np.concatenate((data['X_in'], data['X_out_5'], data['X_out_6']))
    y_tr = np.concatenate((data['y_in'], data['y_out_5'], data['y_out_6']))
    X_tr, y_tr = shuffle(X_tr, y_tr, random_state=0)
    # test outliers
    X_te = data['X_out_7'] 
    y_te = data['y_out_7']
    return X_tr, y_tr, X_te, y_te


def get_fs_dataset(dataset_name: str, seed: int):
    """Load a dataset given its name

    Valid dataset are
    * 'cardio'
    * 'ionosphere'
    * 'letter'
    * 'musk'
    * 'lympho'
    * 'satellite'

    Parameters
    ----------
    dataset_id : str
        Name of the dataset
    seed : int
        Random State

    Returns
    -------
    Tuple
        X, y, contamination

    Raises
    ------
    ValueError
        Invalid Dataset
    """
    if dataset_name in VALID_DATASETS:
        mat = loadmat(str(PACKAGE_PATH / "datasets" / "ufs" / (dataset_name + ".mat")))
        X = mat["X"]
        y = mat["y"].squeeze()
        logger.info(
            "\nLoaded {} dataset: {} samples, {} features.".format(
                dataset_name, X.shape[0], X.shape[1]
            )
        )
    else:
        raise ValueError(f"Invalid Dataset provided: Valids are {VALID_DATASETS}")
    y = y.astype("int")
    contamination = len(y[y == 1]) / len(y)
    logger.info("{:2.2f} percent outliers.".format(contamination * 100))
    X, y = shuffle(X, y, random_state=seed)
    return X, y, contamination


def fs_datasets_hyperparameters(dataset):
    data = {
        "cardio": {"contamination": 0.1, "max_samples": 64, "n_estimators": 150},
        "ionosphere": {"contamination": 0.2, "max_samples": 256, "n_estimators": 100},
        "lympho": {"contamination": 0.05, "max_samples": 64, "n_estimators": 150},
        "letter": {"contamination": 0.1, "max_samples": 256, "n_estimators": 50},
        "musk": {"contamination": 0.05, "max_samples": 128, "n_estimators": 100},
        "satellite": {"contamination": 0.15, "max_samples": 64, "n_estimators": 150},
    }
    return data[dataset]