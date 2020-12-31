import datetime
import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.datasets import load_svmlight_file

TRAIN_FNAME = "../data/train.txt"
TEST_FNAME = "../data/test.txt"

mem = Memory("./mycache")


def get_query_generator(inds, X, y, min_length=1, equal_length=None, 
                            drop_all_zeros=True, only_one_sample=False):
    for index in inds:
        if (equal_length is not None and len(index) == equal_length) or \
            (equal_length is None and len(index) > min_length):
            x = X[index,:].toarray()
            y_ = y[index][:,np.newaxis]
            if drop_all_zeros and np.max(y_) <= 0:
                continue
            yield x, y_
            if only_one_sample:
                break

@mem.cache
def get_data():
    data = load_svmlight_file(TRAIN_FNAME)
    return data

@mem.cache
def get_data_test():
    data = load_svmlight_file(TEST_FNAME)
    return data

@mem.cache
def get_inds():
    qids = []
    with open(TRAIN_FNAME, "r") as f:
        for line in f.readlines():
            s = line.split(" ", 2)
            qids.append(s[:2])
    df_qids = pd.DataFrame(qids, columns=['rank', 'qid'])
    inds = df_qids.groupby("qid").apply(lambda df: df.index.values)
    return inds

@mem.cache
def get_df_qids_test():
    qids = []
    with open(TEST_FNAME, "r") as f:
        for line in f.readlines():
            s = line.split(" ", 2)
            qids.append(s[:2])
    df_qids = pd.DataFrame(qids, columns=['rank', 'qid'])
    return df_qids

@mem.cache
def get_inds_test():
    df_qids = get_df_qids_test()
    inds = df_qids.groupby("qid").apply(lambda df: df.index.values)
    return inds

def get_selected_features(X):
    X = X.toarray()
    ts = X.std(axis=0)
    zero_std_features = ts == 0
    return ~zero_std_features

def get_submit(preds, inds_test):
    df_qids = get_df_qids_test()
    df_list = []
    for index in inds_test:
        pred = preds[:,0][index]
        order = np.argsort(pred)[::-1]
        index_sorted = index[order]
        qid = df_qids['qid'].iloc[index_sorted]
        assert qid.nunique() == 1
        df_list.append(qid)
    df_res = pd.concat(df_list)
    df_res = df_res.reset_index()
    df_res['index'] += 1
    df_res['qid'] = df_res['qid'].str.split(":").apply(lambda x: x[1])
    df_res = df_res.rename(columns={"qid": "QueryId", "index": "DocumentId"})
    return df_res[["QueryId","DocumentId"]]