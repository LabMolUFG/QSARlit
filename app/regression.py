# %%
# Importing packages 
import rdkit
from rdkit import Chem, DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import PandasTools
from rdkit.Chem import MACCSkeys


import numpy as np
from numpy import sqrt
from numpy import argmax

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from pandas import DataFrame

import lightgbm as lgb

from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import accuracy_score, cohen_kappa_score, matthews_corrcoef, roc_curve, precision_recall_curve, roc_auc_score, make_scorer
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import auc as mauc
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_digits
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_predict

from imblearn.metrics import geometric_mean_score

from skopt import BayesSearchCV

# %%
#defining functions:

def getNeighborsDitance(trainingSet, testInstance, k):
    neighbors_k=metrics.pairwise.pairwise_distances(trainingSet, Y=testInstance, metric='dice', n_jobs=1)
    neighbors_k.sort(0)
    similarity= 1-neighbors_k
    return similarity[k-1,:]

#5-fold-cross-val
def cros_val(x,y,classifier):
    probs_classes = []
    #indexes = []
    y_test_all = []
    AD_fold =[]
    distance_train_set =[]
    distance_test_set = []
    y_pred_ad=[]
    y_exp_ad =[]

    for train_index, test_index in kf.split(x, y):
        clf = classifier # model with best parameters
        X_train_folds = x[train_index] # descritors train split
        y_train_folds = np.array(y)[train_index.astype(int)] # label train split
        X_test_fold = x[test_index] # descritors test split
        y_test_fold = np.array(y)[test_index.astype(int)] # label test split


        clf.fit(X_train_folds, y_train_folds) # train fold
        y_pred = clf.predict_proba(X_test_fold) # test fold
        probs_classes.append(y_pred) # all predictions for test folds
        y_test_all.append(y_test_fold) # all folds' labels 
        #   indexes.append(test_index) # all tests indexes

        # DA
        k= int(round(pow((len(y)) ,1.0/3), 0))
        distance_train = getNeighborsDitance(X_train_folds, X_train_folds, k)
        distance_train_set.append(distance_train)
        distance_test = getNeighborsDitance(X_train_folds, X_test_fold, k)
        distance_test_set.append(distance_test)
        #Dc = np.average(distance_train_set)-(1*np.std(distance_train_set))
        Dc=0.5
        for i in range(len(X_test_fold)):
            ad=0
            if distance_test_set[0][i] >= Dc:
                ad = 1
            AD_fold.append(ad)


    # Get predictions of each fold
    fold_1_pred = (probs_classes[0][:, 1] > 0.5).astype(int)
    fold_2_pred = (probs_classes[1][:, 1] > 0.5).astype(int)
    fold_3_pred = (probs_classes[2][:, 1] > 0.5).astype(int)
    fold_4_pred = (probs_classes[3][:, 1] > 0.5).astype(int)
    fold_5_pred = (probs_classes[4][:, 1] > 0.5).astype(int)

    # Get experimental values of each fold
    fold_1_exp = y_test_all[0]
    fold_2_exp = y_test_all[1]
    fold_3_exp = y_test_all[2]
    fold_4_exp = y_test_all[3]
    fold_5_exp = y_test_all[4]

    bacc1 = metrics.balanced_accuracy_score(fold_1_exp, fold_1_pred) # balanced accuracy fold 1
    bacc2 = metrics.balanced_accuracy_score(fold_2_exp, fold_2_pred) # balanced accuracy fold 2
    bacc3 = metrics.balanced_accuracy_score(fold_3_exp, fold_3_pred) # balanced accuracy fold 3
    bacc4 = metrics.balanced_accuracy_score(fold_4_exp, fold_4_pred) # balanced accuracy fold 4
    bacc5 = metrics.balanced_accuracy_score(fold_5_exp, fold_5_pred) # balanced accuracy fold 5
    print("Balanced accuracy (fold 1) = ", bacc1)
    print("Balanced accuracy (fold 2) = ", bacc2)
    print("Balanced accuracy (fold 3) = ", bacc3)
    print("Balanced accuracy (fold 4) = ", bacc4)
    print("Balanced accuracy (fold 5) = ", bacc5)

    probs_classes = np.concatenate(probs_classes)    
    y_experimental = np.concatenate(y_test_all)
    # Uncalibrated model predictions
    pred = (probs_classes[:, 1] > 0.5).astype(int)
    for i in range(len(AD_fold)):
        if AD_fold[i] == 1:
            y_pred_ad.append(pred[i])
            y_exp_ad.append(y_experimental[i])
    
            
    
    return(pred, y_experimental, probs_classes, AD_fold, y_pred_ad, y_exp_ad)

#CALIBRATION
def calibration_curve_plot(probs_classes, y_exp):
    # keep probabilities for the positive outcome only
    probs = probs_classes[:, 1]
    # reliability diagram
    fop, mpv = calibration_curve(y_exp, probs, n_bins=10)
    # plot perfectly calibrated
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot model reliability
    plt.plot(mpv, fop, marker='.')
    plt.show()

def calibration_threshold_roc(probs_classes, y_exp):
    # keep probabilities for the positive outcome only
    yhat = probs_classes[:, 1]
    # calculate roc curves
    fpr, tpr, thresholds = roc_curve(y_exp, yhat)
    # calculate the g-mean for each threshold
    gmeans = sqrt(tpr * (1-fpr))
    # locate the index of the largest g-mean
    ix = argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    # plot the roc curve for the model
    plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label='RF')
    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    # show the plot
    plt.show()
    threshold_roc = thresholds[ix]
    return(threshold_roc)

def calibration_threshold_prc(probs_classes, y_exp):
    # keep probabilities for the positive outcome only
    yhat = probs_classes[:, 1]
    # calculate precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_exp, yhat)
    # convert to f score
    fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ix = argmax(fscore)
    print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
    threshold_prc = thresholds[ix]
    return(threshold_prc)

#STATISTICS
def calc_statistics(y,pred):
    # save confusion matrix and slice into four pieces
    confusion = confusion_matrix(y, pred)
    #[row, column]
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    
    # Plot confusion
    #plt.figure(figsize=(5,5))
    #sns.heatmap(confusion, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r');
    #plt.ylabel('Actual label');
    #plt.xlabel('Predicted label');
    #title = "Confusion matrix"
    #plt.title(title, size = 15);
    
    # calc statistics
    classification_error = 1 - accuracy_score(y, pred) #Classification error or misclassification rate
    accuracy = accuracy_score(y, pred) #accuracy
    mcc = matthews_corrcoef(y, pred) #mcc
    kappa = cohen_kappa_score(y, pred) #kappa
    sensitivity = recall_score(y, pred) #Sensitivity
    specificity = TN / (TN + FP) #Specificity
    false_positive_rate = FP / float(TN + FP) #False positive rate (alfa)
    false_negative_rate = FN / float(TP+FN) #False negative rate (beta)
    precision = TP / float(TP + FP) #Precision
    positive_pred_value = TP / float(TP + FP) #PPV
    negative_pred_value = TN / float(TN + FN) #NPV
    auc = roc_auc_score(y, pred) #AUC
    bacc = balanced_accuracy_score(y, pred) # balanced accuracy
    f1 = f1_score(y, pred) # F1-score

    print("Accuracy = ", accuracy)
    print("MCC = ", mcc)
    print("Kappa = ", kappa)
    print("Sensitivity = ", sensitivity)
    print("Specificity = ", specificity)
    print("Precision = ", precision)
    print("PPV = ", positive_pred_value)
    print("NPV = ", negative_pred_value)
    print("False positive rate = ", false_positive_rate)
    print("False negative rate = ", false_negative_rate)
    print("AUC = ",roc_auc_score(y, pred))
    print("Classification error = ", classification_error)
    print("Balanced accuracy = ", bacc)
    print("F1-score = ", f1)
    
    #converting calculated metrics into a pandas dataframe to compare all models at the final
    statistics = pd.DataFrame({'Bal-acc': bacc, "Sensitivity": sensitivity, "Specificity": specificity,"PPV": positive_pred_value, 
           "NPV": negative_pred_value, 'Kappa': kappa, 'AUC': auc, 'MCC': mcc, 'Accuracy': accuracy, 
           "Classification error": classification_error,"False positive rate": false_positive_rate, 
           "False negative rate": false_negative_rate, "Precision": precision, 'F1-score': f1,}, index=[0])
    return(statistics)

# %%
# Reading molecules and activity (0 and 1) from SDF
fname = r"D:\\hergproject_new\\postcurationscript_data\\CB_CHO_IC50.csv"
data = pd.read_csv(fname)
data = data[0:500]
mols = [Chem.MolFromSmiles(smile) for smile in data['SMILES']]

# %%
# generate binary maccs fingerprint
fp = [MACCSkeys.GenMACCSKeys(x) for x in mols]

def rdkit_numpy_convert(fp):
    output = []
    for f in fp:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(f, arr)
        output.append(arr)
    return np.asarray(output)

x = rdkit_numpy_convert(fp)
y = [num for num in data['pIC50 (uM)']]

# %%
#randomly select 20% of compounds as external set
x, x_ext, y, y_ext = train_test_split(x, y, test_size=0.20, random_state=42)

# %%
x_check = pd.DataFrame(x)
y_check = pd.DataFrame(y)
print("Number of compounds and descriptors in training set:", x_check.shape)
print("Number of compounds and target variables in training set:", y_check.shape)

# %%
x_ext_check = pd.DataFrame(x_ext)
y_ext_check = pd.DataFrame(y_ext)

print("Number of compounds and descriptors in external set:", x_ext_check.shape)
print("Number of compounds and target variables in external set:", y_ext_check.shape)

# %%
kf = KFold(n_splits = 5, shuffle = True, random_state = 42)

# %% [markdown]
# ## LGBM REGRESSOR

# %%
lgbr = lgb.LGBMRegressor(objective = "regression", n_jobs = -1, random_state = 42, learning_rate = 0.1, n_estimators = 200)

# %%
opt_lgbr = BayesSearchCV(lgbr, 
                     {
                        "max_depth": [3, 13],
                        "num_leaves": [20, 200],
                       "min_child_samples": [7, 75],
                       "colsample_bytree": [0.25, 1],
                       "subsample": [0.25, 1],
                        "subsample_freq": [1, 50],
                        "reg_alpha": [0, 1],
                       "reg_lambda": [0, 1],
                       "min_split_gain": [0, 0.5]
                   },
                   n_iter = 150,
                   cv = kf,
                  n_jobs = -1,
                  scoring = "neg_root_mean_squared_error",
                  random_state = 42
                 )


opt_lgbr.fit(x, y)

# %%
scoring = ['max_error', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2']
scores = cross_validate(opt, x, y, cv=5, scoring=scoring)

# %%
print("MAX ERROR %0.2f" % (scores['test_max_error'].mean()*-1))
print("MAE %0.2f" % (scores['test_neg_mean_absolute_error'].mean()*-1))
print("MSE %0.2f" % (scores['test_neg_mean_squared_error'].mean()*-1))
print("RMSE %0.2f" % (scores['test_neg_root_mean_squared_error'].mean()*-1))
print("R2 %0.2f" % (scores['test_r2'].mean()))

# %% [markdown]
# ## RANDOM FOREST REGRESSOR

# %%
scorer = make_scorer(geometric_mean_score)

# log-uniform: understand as search over p = exp(x) by varying x
opt_rf = BayesSearchCV(
    RandomForestRegressor(),
    {'max_features': ['auto', 'sqrt'],
    'n_estimators': [100, 1000],
    "max_depth": [2, 100],
    'min_samples_leaf': [1,20], 
    'min_samples_split': [2, 20]
    },
    n_iter=30, # Number of parameter settings that are sampled
    cv=kf,
    scoring = "neg_root_mean_squared_error",
    verbose=0,
    refit= True, # Refit the best estimator with the entire dataset.
    random_state=42, 
    n_jobs = -1
)

opt_rf.fit(x, y)

# %%
scoring = ['max_error', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2']
scores = cross_validate(opt_rf, x, y, cv=5, scoring=scoring)

# %%
print("MAX ERROR %0.2f" % (scores['test_max_error'].mean()*-1))
print("MAE %0.2f" % (scores['test_neg_mean_absolute_error'].mean()*-1))
print("MSE %0.2f" % (scores['test_neg_mean_squared_error'].mean()*-1))
print("RMSE %0.2f" % (scores['test_neg_root_mean_squared_error'].mean()*-1))
print("R2 %0.2f" % (scores['test_r2'].mean()))

# %% [markdown]
# ## SUPPORT VECTOR REGRESSOR

# %%
scorer = make_scorer(geometric_mean_score)

# log-uniform: understand as search over p = exp(x) by varying x
opt_svr = BayesSearchCV(
    SVR(),
    {
        'C': (1e-6, 1e+6, 'log-uniform'),
        'gamma': (1e-6, 1e+1, 'log-uniform'),
        'kernel': ['rbf'],  # categorical parameter | ['linear', 'poly', 'rbf'] to test all kernels
    },
    n_iter=30, # Number of parameter settings that are sampled
    cv=kf,
    scoring = "neg_root_mean_squared_error",
    refit = True, # Refit the best estimator with the entire dataset.
    random_state=42,
    n_jobs = -1
)

opt_svr.fit(x, y)

# %%
scoring = ['max_error', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2']
scores = cross_validate(opt_svm, x, y, cv=5, scoring=scoring)

# %%
print("MAX ERROR %0.2f" % (scores['test_max_error'].mean()*-1))
print("MAE %0.2f" % (scores['test_neg_mean_absolute_error'].mean()*-1))
print("MSE %0.2f" % (scores['test_neg_mean_squared_error'].mean()*-1))
print("RMSE %0.2f" % (scores['test_neg_root_mean_squared_error'].mean()*-1))
print("R2 %0.2f" % (scores['test_r2'].mean()))

# %%



