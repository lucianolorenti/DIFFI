import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def plot_ranking_glass(ord_idx, title):
    sns.set(style='darkgrid')
    id2feat = {0:'RI', 1:'Na', 2:'Mg', 3:'Al', 4:'Si', 5:'K', 6:'Ca', 7:'Ba', 8:'Fe'}
    x_ticks = [r'$1^{st}$', r'$2^{nd}$', r'$3^{rd}$', r'$4^{th}$', r'$5^{th}$', r'$6^{th}$', r'$7^{th}$', r'$8^{th}$', r'$9^{th}$']
    num_feats = ord_idx.shape[1]
    features = np.arange(num_feats)
    ranks = np.arange(1, num_feats+1)
    rank_features = {r: [list(ord_idx[:,r-1]).count(f) for f in features] for r in ranks}
    df = pd.DataFrame(rank_features)
    df_norm = df.transform(lambda x: x/sum(x))
    df_norm['Feature ID'] = features
    df_norm['Feature'] = df_norm['Feature ID'].map(id2feat)
    sns.set(style='darkgrid')
    df_norm.drop(['Feature ID'], inplace=True, axis=1)
    df_norm.set_index('Feature').T.plot(kind='bar', stacked=True)
    locs, labels = plt.xticks()
    plt.ylim((0, 1.05))
    plt.xticks(locs, x_ticks, rotation=0)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', title='Feature', ncol=5, mode='expand', borderaxespad=0.)
    plt.xlabel('Rank')
    plt.ylabel('Normalized count')
    plt.title(title, y=1.3)


def plot_ranking_syn(ord_idx, title):
    sns.set(style='darkgrid')
    id2feat = {0:r'$f_1$', 1:r'$f_2$', 2:r'$f_3$', 3:r'$f_4$', 4:r'$f_5$', 5:r'$f_6$'}
    x_ticks = [r'$1^{st}$', r'$2^{nd}$', r'$3^{rd}$', r'$4^{th}$', r'$5^{th}$', r'$6^{th}$']
    num_feats = ord_idx.shape[1]
    features = np.arange(num_feats)
    ranks = np.arange(1, num_feats+1)
    rank_features = {r: [list(ord_idx[:,r-1]).count(f) for f in features] for r in ranks}
    df = pd.DataFrame(rank_features)
    df_norm = df.transform(lambda x: x/sum(x))
    df_norm['Feature ID'] = features
    df_norm['Feature'] = df_norm['Feature ID'].map(id2feat)
    sns.set(style='darkgrid')
    df_norm.drop(['Feature ID'], inplace=True, axis=1)
    df_norm.set_index('Feature').T.plot(kind='bar', stacked=True)
    locs, labels = plt.xticks()
    plt.ylim((0, 1.05))
    plt.xticks(locs, x_ticks, rotation=0)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', title='Feature', ncol=6, mode='expand', borderaxespad=0.)
    plt.xlabel('Rank')
    plt.ylabel('Normalized count')
    plt.title(title, y=1.3)


def plot_new_outliers_syn(X_xaxis, X_yaxis, X_bisec, title):
    sns.set(style='darkgrid')
    plt.scatter(X_xaxis[:,0], X_xaxis[:,1], cmap='Blues')
    plt.scatter(X_yaxis[:,0], X_yaxis[:,1], cmap='Greens')
    plt.scatter(X_bisec[:,0], X_bisec[:,1], cmap='Oranges')
    plt.title(title)
