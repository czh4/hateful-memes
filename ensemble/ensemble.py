import os
import re
import glob
import argparse
import pandas as pd
import numpy as np
from scipy.special import logit, expit

from sklearn import linear_model
from sklearn.metrics import roc_auc_score, accuracy_score

parser = argparse.ArgumentParser(description='Enter path and ensemble method')
parser.add_argument('--path', type=str, help='path of val files')
parser.add_argument('--method', type=str, help='ensemble method, avg or linear')
parser.add_argument('--phase', type=int, help='phase 1 or 2, enter integer')

args = parser.parse_args()

## build ensemble model based on dev_unseen set

dev_df = pd.read_json(os.path.join(args.path,'dev_unseen.jsonl'), lines=True)

# naming of val_files should be <model_name>_<method>_<auc>_val.csv
# for example, vilbert_original_7075_val.csv
val_files = [file for folder,_,_ in os.walk(args.path) for file in glob.glob(os.path.join(folder, '*val.csv'))]

for val in val_files:
    df = pd.read_csv(val)
    name = re.split('_|\.', val)[-3] # column name based on auc score
    dev_df = dev_df.merge(df[['id','proba']], on='id').rename(columns={'proba':'proba_'+name})

# proba needs to convert ot logit
X = dev_df.iloc[:,4:].apply(lambda x : logit(x))
y = dev_df.label

# load corresponding test predictions
# naming of test files should be similar to val files
if args.phase == 1:
    test_df = pd.read_json(os.path.join(args.path,'test_seen.jsonl'), lines=True)
    test_files = [file for folder,_,_ in os.walk('preds') for file in glob.glob(os.path.join(folder, '*test_seen.csv'))]
elif args.phase == 2:
    test_df = pd.read_json(os.path.join(args.path,'test_unseen.jsonl'), lines=True)
    test_files = [file for folder,_,_ in os.walk('preds') for file in glob.glob(os.path.join(folder, '*test_unseen.csv'))]

for test in test_files:
    df = pd.read_csv(test)
    name = re.split('_|\.', test)[-4] # column name based on auc score in val_unseen set
    test_df = test_df.merge(df[['id','proba']], on='id').rename(columns={'proba':'proba_'+name})

X_test = test_df.iloc[:,3:].apply(lambda x : logit(x))


if args.method == 'linear':
    # ensemble method 1: build an logistic regression model
    clf = linear_model.LogisticRegression(fit_intercept=False, solver='liblinear')
    clf.fit(X,y)
    pred = clf.predict_proba(X_test)[:,1]
elif args.method == 'avg':
    # ensemble method 2: mean of logits
    pred = expit(X_test.mean(axis=1))

test_df['proba'] = pred
test_df['label'] = test_df['proba'].apply(lambda x : 1 if x>=0.5 else 0)

test_df[['id','proba','label']].to_csv(os.path.join(path,'ensemble.csv'), index=False)