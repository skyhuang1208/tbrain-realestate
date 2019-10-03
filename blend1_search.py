from __future__ import print_function
# ## Import Libraries
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score


########## **********  Various functions which I didn't do look  ********** ##########


# Define the gini metric - from https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703#5897
def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

def gini_normalized(actual, pred):
    return gini(actual, pred) / gini(actual, actual)

def cal_hit_rate(y_true, y_pred):
    rate = np.mean(np.abs(y_pred - y_true) <= 0.1 * y_true)
    mape = np.mean(np.abs(y_pred - y_true) / y_true)
    return np.round(rate, decimals=4)*10000 + (1-mape)

# Create an XGBoost-compatible Gini metric
def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return [('gini', gini_score)]

def gini_xgb_min(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return [('gini', -1*gini_score)]

# Some functions from Andy: https://www.kaggle.com/aharless/xgboost-cv-lb-284


import time
t0= time.time()

is_filein = False
if len(sys.argv) == 3 and sys.argv[1] == 'FILEIN':
    print('Reading arguments from file {}'.format(sys.argv[2]))
    is_filein = True
elif len(sys.argv) <5 or not sys.argv[1].isdigit() or int(sys.argv[1])+4 != len(sys.argv):
    print()
    print('Usage: %s _N_models_ [actual] [predict 1] [predict 2] [predict 3] ... [out_file_name]' % sys.argv[0])
    print('   or  %s  FILEIN input_params_file' % sys.argv[0])
    print()

    raise Exception('Need input args')


print('\nLinear Combination Model (blend). Search coeffs\n')
isgini= input('Choose metrics to be log loss or gini (0: log loss, 1: gini, 2: auc, 3: hit-ratio)\n')
mode=   input('Choose your search mode (0: grid only, 1: grid+rand, 2: rand_init+rand, 3: read_coeffs+rand)\n')
islog=  input('Do you want to combine in log (log1p -> exp1m) (0: no, 1: yes)\n')

col_id = input('\nEnter the col of id\n')
col_target = input('\nEnter the col of target\n')
col_target_predict = input('\nEnter the col of predicted target\n')

Nmds = -1
file_actual = ''
files_pred = []
file_output = ''
if is_filein:
    with open(sys.argv[2], 'r') as INARGS:
        for i, line in enumerate(INARGS):
            if i==0:   Nmds = int(line.strip())
            elif i==1: file_actual = line.strip()
            else:      files_pred.append(line.strip())
    file_output = input('\nEnter the output log file name\n')
else:
    Nmds= int(sys.argv[1])
    files_pred = sys.argv[2:-1]
    file_output = sys.argv[-1]


########## READ FILES ##########
actual = pd.read_csv(file_actual).sort_values(by=col_id)
y_actual = actual[col_target].values
y_pred = []
for i in range(Nmds):
    model = pd.read_csv(files_pred[i]).sort_values(by=col_id)
    if (actual[col_id].values != model[col_id].values).any():
        exit(f'id not same: {files_pred[i]}')
    y_pred.append(model[col_target_predict].values)


### PARS ###
Y_SIZE= y_actual.shape[0]
NUM_RATIO= 5            # N of possible ratio
GRID_SIZE= 1./NUM_RATIO # grid size (period)
N_FINE= 1000
### PARS ###


### INIT VARS ###
results= {}
coeff= [0 for _ in range(Nmds)]
coeff[-1]= NUM_RATIO
### INIT VARS ###


########## GRID SEARCH ##########
if mode == '0' or mode == '1':
    while True:
        # Calculate gini score
        y_blend= np.zeros(Y_SIZE)
        for i, c in enumerate(coeff):
            ratio= c * GRID_SIZE
            if islog=='1':
                y_blend= np.add( ratio * np.log1p(y_pred[i]) , y_blend )
            else:
                y_blend= np.add( ratio * y_pred[i] , y_blend )
    
        if islog=='1':
            y_blend= np.clip(np.expm1(y_blend), 0, None)

        if isgini=='0':
            loss_score = log_loss(y_actual, y_blend)
            results[tuple(np.multiply(coeff, GRID_SIZE))]= loss_score
        elif isgini=='1':
            gini_score = gini_normalized(y_actual, y_blend)
            results[tuple(np.multiply(coeff, GRID_SIZE))]= gini_score
        elif isgini=='2':
            auc_score = roc_auc_score(y_actual, y_blend)
            results[tuple(np.multiply(coeff, GRID_SIZE))]= auc_score
        elif isgini=='3':
            hit_score = cal_hit_rate(y_actual, y_blend)
            results[tuple(np.multiply(coeff, GRID_SIZE))]= hit_score
        else:
            exit('isgini is 0, 1, 2, or 3')

        # Make next coeff set
        sumup= sum(coeff[:-1])
        if sumup > NUM_RATIO:
            exit('Err: sum coeff[:-1] too large: %f' % sumup)
        elif sumup == NUM_RATIO:
            if coeff[-2]==NUM_RATIO: break # done searching

            for i, c in enumerate(coeff[:-1]):
                if c != 0:
                    coeff[i]= 0
                    coeff[i+1] +=1
                    if i==len(coeff)-3: print('\r', coeff[-2], '/', NUM_RATIO, end='')
                    break
        else: 
            coeff[0] += 1

        coeff[-1]= NUM_RATIO - sum(coeff[:-1])

    if isgini=='0': results= sorted(results.items(), key= lambda x:x[1], reverse= False)
    else:           results= sorted(results.items(), key= lambda x:x[1], reverse= True)
elif mode == '2':
    coeff = np.random.rand(Nmds)
    total = coeff.sum()
    coeff = [c/total for c in coeff]
    print('Randomly init coeffs:')
    print(coeff)
    results = [[coeff, None]]
    GRID_SIZE = float(input('Set up grid size (0-1)'))
elif mode == '3':
    coeff_in = input('Enter coeffs: one line with format either c1, c2, ... or [c1, c2, ...]')
    if '-' in coeff_in: raise Exception('cannot have neg coeff')
    coeff_in = coeff_in.replace('[','').replace(']','')
    if ',' in coeff_in: coeff = list(map(float, coeff_in.split(',')))
    else:               coeff = list(map(float, coeff_in.split()))
    if abs(1 - sum(coeff)) > 0.01: raise Exception('Coeffs need to sum up to 1')
    results = [[coeff, None]]
    GRID_SIZE = float(input('Set up grid size (0-1)'))
else:
    raise Exception('Mode need to be 1, 2, 3, or 4 but get: {}'.format(mode))


########## FINE RANDOM SEARCH ##########
if mode != '0':
    print('\nNow doing random fine-search...\n')

    ##### Generate coeffs randomly deviated from best
    def rand_coeffs(coeffs):
        Nc= len(coeffs)

        sigma = 0.5 * GRID_SIZE
        def gen_rand():
            rand= sigma * np.random.randn()
            return rand if rand > 0. else 0
        
        while True:
            ipick= np.random.randint(Nc) # pick one not random (force sum to 1.)
            new= np.zeros(Nc)
            total= 0.
            for i, c in enumerate(coeffs):
                if i==ipick: continue
                new[i]= c + gen_rand()
                total += new[i]
            if total <= 1.:
                new[ipick]= 1. - sum(new)
                return new


    ##### Iteratively search for better coeffs #####
    best= list(results[0]) # [0]: coeffs, [1]: score
    results_fine= {}
    for Niter in range(N_FINE): # start random search
        if Niter%10 == 0: print('\r%d/%d' % (Niter, N_FINE), end='')

        # make new coeffs
        if Niter == 0: 
            new_coeffs = best[0]
        else:
            new_coeffs = rand_coeffs(best[0])
        y_blend= np.zeros(Y_SIZE)
        for i, c in enumerate(new_coeffs):
            if islog=='1':
                y_blend= np.add( c * np.log1p(y_pred[i]) , y_blend )
            else:
                y_blend= np.add( c * y_pred[i] , y_blend )
    
        if islog=='1':
            y_blend= np.clip(np.expm1(y_blend), 0, None)

        
        # evaluate
        if isgini=='0':
            loss_score = log_loss(y_actual, y_blend)
            if best[1] is None or loss_score < best[1]:
                best= [new_coeffs, loss_score]
                results_fine[tuple(new_coeffs)]= loss_score
                print('found better score {} at Niter {}.'.format(loss_score, Niter))
        elif isgini=='1':
            gini_score = gini_normalized(y_actual, y_blend)
            if best[1] is None or gini_score > best[1]:
                best= [new_coeffs, gini_score]
                results_fine[tuple(new_coeffs)]= gini_score
                print('found better score {} at Niter {}.'.format(gini_score, Niter))
        elif isgini=='2':
            auc_score = roc_auc_score(y_actual, y_blend)
            if best[1] is None or auc_score > best[1]:
                best= [new_coeffs, auc_score]
                results_fine[tuple(new_coeffs)]= auc_score
                print('found better score {} at Niter {}.'.format(auc_score, Niter))
        elif isgini=='3':
            hit_score = cal_hit_rate(y_actual, y_blend)
            if best[1] is None or hit_score > best[1]:
                best= [new_coeffs, hit_score]
                results_fine[tuple(new_coeffs)]= hit_score
                print('found better score {} at Niter {}.'.format(hit_score, Niter))
        else:
            exit('isgini is 0, 1, 2 or 3')

    
    if isgini=='0':
        results_fine= sorted(results_fine.items(), key= lambda x:x[1], reverse= False)
    else:
        results_fine= sorted(results_fine.items(), key= lambda x:x[1], reverse= True)


########## Output results ##########
OFILE= open(file_output, 'w')
if mode != '0':
    print('# Random Fine Tune Results', file=OFILE)
    for key, val in results_fine: 
        for k in key: print('%.5f' % k, file=OFILE, end= ' ')
        print(val, file=OFILE)
    print(file=OFILE)

print('# Grid Search Results', file=OFILE)
for key, val in results:
    print(*key, val, file=OFILE)

print('\nTime spent: ', time.time()-t0)
