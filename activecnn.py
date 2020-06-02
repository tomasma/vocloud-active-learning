import csv
import json

import numpy as np
from scipy.stats import entropy

from active_cnn import data
from active_cnn import model
from active_cnn import preprocessing

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def activeCnn(pool_csv,training_set_csv,n_classes,cat_list,candidate_classes,batch_size,random_sample_size):
    # load the parameters from config file
    CONFIG = "config.json"
    with open(CONFIG) as f:
        config = json.load(f)

        # input files:
        # labelled training set
        # TRAINING_SET_CSV = config["training_set_csv"]
        TRAINING_SET_CSV = training_set_csv
        # unlabelled target spectra
        POOL_CSV = pool_csv

        # output files:
        # randomly selected spectra for true positive rate estimation
        try:
            PERFORMANCE_ESTIMATION_CSV = config["performance_estimation_csv"]
        except:
            PERFORMANCE_ESTIMATION_CSV = "perf-est.csv"
        # spectra for oracle selected according to information entropy
        try:
            ORACLE_CSV = config["oracle_csv"]
        except:
            ORACLE_CSV = "oracle.csv"
        try:
            CANDIDATES_CSV = config["candidates_csv"]
        except:
            CANDIDATES_CSV = "candidates.csv"
        # parameters:
        # size of sample for performance estimation
        #RANDOM_SAMPLE_SIZE = config["random_sample_size"]
        RANDOM_SAMPLE_SIZE = random_sample_size
        # number of spectra in a batch for labelling of oracle
        #BATCH_SIZE = config["batch_size"]
        BATCH_SIZE = batch_size

    # load correctly data
    ids, X = data.get_pool(POOL_CSV)
    data_tr = data.get_training_set(TRAINING_SET_CSV)
    ids_tr, X_tr, y_tr = data_tr


    # learning stage:
    # SMOTE balancing
    X_tr_bal, y_tr_bal = preprocessing.balance(X_tr, y_tr)

    # CNN training
    cnn = model.get_model(n_classes)
    model.train(cnn, X_tr_bal, y_tr_bal,n_classes)

    # classification of unlabelled samples
    y_pred = model.predict(cnn, X)
    # get labels
    labels = np.argmax(y_pred, axis=1)

    # performance estimation sample:
    # take only positive examples: emissions and double-peaks
    trg_idx = np.arange(labels.shape[0])[labels != 0]
    rnd_idx = np.random.choice(trg_idx, size=RANDOM_SAMPLE_SIZE, replace=False)

    # uncertainty sampling of batch of spectra for oracle:
    # compute entropies
    entropies = entropy(y_pred.T)

    # write PERFORMANCE_ESTIMATION_CSV
    with open(PERFORMANCE_ESTIMATION_CSV, 'w', newline='') as f:
       writer = csv.writer(f)
       writer.writerows(zip(ids[rnd_idx], labels[rnd_idx],entropies[rnd_idx]))

    batch_idx = np.argsort(entropies)[-BATCH_SIZE:]
    #print(batch_idx)
    #print(entropies[batch_idx])

    # write CSV
    with open(ORACLE_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(zip(ids[batch_idx], labels[batch_idx], entropies[batch_idx]))

    with open(CANDIDATES_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        #writer.writerows(zip(ids[labels>0], labels[labels>0], entropies[labels>0]))
        for i in range(len(labels)):
           if cat_list[labels[i]] in candidate_classes:
              writer.writerow([ids[i], labels[i], entropies[i]])
    # calculate statistics
    statistics = []
    for i in range(n_classes):
        statistics.append(len(labels[labels==i]))
        #print('Statistics '+str(i)+': '+str(statistics[i]))

    #return statistics,ids,labels
    return statistics