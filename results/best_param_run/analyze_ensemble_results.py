# plot results computed by COMPARISON.py
# code is ugly, but works...

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sns
import pandas
import glob
import os
import Utils
from statsmodels import robust
import prettyplotlib as ppl
import string
import itertools
import scipy.io as io
from scipy.stats import spearmanr

def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Calculates the quadratic weighted kappa
    quadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator



def plot_predictions_advanced(model,X,X_feat,tokens,real_pred):

    MAX_ROW_LENGTH = 25 # tokens

    try:
        weights = (model.best_estimator_._intercept_,model.best_estimator_.coef_.flatten())
    except:
        try:
            weights = (model._intercept_,model.coef_.flatten())
        except:
            weights = (model.intercept_, model.coef_.flatten())

    assert len(X_feat)==len(weights[1])

    ind=[[],[],[]]
    terms = {}
    for k,feat in enumerate(X_feat):
        if feat.find('emb300_')>-1:
            ind[0].append(k)
        elif feat.find('term=')>-1:
            ending = feat.find(',type=')
            term = feat[5:ending]
            #if term not in terms:
            terms[term] = [weights[1][k]]
            #else:
            #    terms[term].append(weights[1][k])
            ind[1].append(k)
        else:
            ind[2].append(k)

    #ind[2]=np.array(ind[2])

    x_tokens = np.zeros(len(tokens['NORMAL']))
    #x_tokens_type = ['' for x in x_tokens]

    K = len(x_tokens)
    for k in range(K):
        for k1,typ in enumerate(('LEMMA','NORMAL','POS')):
            term=[]
            term.append((1,tokens[typ][k]))
            if k<(K-1):
                term.append((2,' '.join(tokens[typ][k:(k+2)])))
            if k<(K-2):
                term.append((3,' '.join(tokens[typ][k:(k+3)])))
            term = list(term)
            for kkk in range(len(term)):
                if term[kkk][1] in terms:
                    for w in terms[term[kkk][1]]:
                        x_tokens[k:(k+term[kkk][0])] += w

    pred = np.zeros(4)
    pred.fill(np.nan)
    for k in (0,1,2):
        w = [np.take(weights[1], ind[k]),np.take(X, ind[k])]
        if k==1:
            w1=w
            w2=np.take(X_feat, ind[k])
        if k==2:
            handgrafted = [(np.hstack((w1[0],w[0])),np.hstack((w1[1],w[1]))),
                           np.hstack((w2,np.take(X_feat, ind[k])))]

        pred[k] = np.sum(w[0]*w[1])+weights[0]/3.0

    final_pred = np.sum(pred[0:3])

    assert np.abs(final_pred-real_pred)<1e-5

    mat = np.zeros((100,MAX_ROW_LENGTH))
    mat.fill(np.nan)

    K = len(x_tokens)
    row = 0
    col = 0
    for k in range(K):
        col+=1
        skip_row=False
        if tokens['NORMAL'][k]=='END_OF_PARAGRAPH':
            skip_row=True
        if col>MAX_ROW_LENGTH-1:
            row+=1
            col=0
        mat[row,col] = x_tokens[k]
        if skip_row:
            row+=1
            col=0

    #mat = mat[0:(row+1),:]

    #cmap = colors.ListedColormap(['red', 'blue'])
    #bounds = np.linspace(np.minimum(x_tokens)-1e-6,np.minimum(x_tokens)+1e-6,50)
    #norm = colors.BoundaryNorm(bounds,cmap.N)

    # plt.close()
    # fig = plt.figure(num=1)
    # DPI = float(fig.dpi)
    # fig.set_size_inches(900/DPI, 900/DPI)
    #
    # cmap = ppl.brewer2mpl.get_map('RdBu','diverging',11).mpl_colormap
    #
    # ppl.pcolormesh(fig,plt.gca(), mat,
    #                xticklabels=[],
    #                yticklabels=[],cmap=cmap)
    # #heatmap(mat, [],[],ax=plt.gca())
    # plt.show(block=None)
    return mat,pred,handgrafted

def clean_features(arr):
    for k,s in enumerate(arr):
        s = s.replace('END_OF_PARAGRAPH', '¤')
        if s.find('term=') > -1:
            s = s.replace('term=', '%i-g: ' % (s.count(' ') + 1))
        arr[k]=s
    return arr

def get_method(string):
    s='\'Algorithm\': (\''
    ind1 = string.find(s)
    assert ind1>-1
    ind1+=len(s)
    ind2 = string[ind1:].find('\'')
    return string[ind1:(ind1+ind2)]

def plot_predictions():
    plt.close()
    fig = plt.figure(num=1)
    DPI = float(fig.dpi)
    fig.set_size_inches(900/DPI, 900/DPI)
    plt.scatter(y_real,y_pred,1.5)
    x = (np.minimum(y_real.min(),y_pred.min()),np.maximum(y_real.max(),y_pred.max()))
    dx=(x[1]-x[0])*0.05
    x = x + dx*np.array((-1,1))
    plt.plot(x,x,color='black')
    #plt.show()

    # add some text for labels, title and axes ticks
    ax = fig.get_axes()[0]
    ax.set_ylabel('Prediction',fontsize=20)
    ax.set_xlabel('Real',fontsize=20)
    ax.set_title('Ratings for \'%s\'' % dimension)
    # ax.set_xticks(x_ind)
    # ax.set_xticklabels(x_label,rotation=-60,ha='left',fontsize=4)
    ax.tick_params(axis='x', which='both', labelsize=18)
    ax.tick_params(axis='y',labelsize=18)

    ax.text(x[0]+3.2*dx,x[1]-6.5*dx,r'$R^2 = %0.2f$' % R2,fontdict={'fontsize': 22})

    #ax.set_xlim([x_ind[0]-3,x_ind[-1]+3])
    ax.autoscale(enable=True, axis='both', tight=True)

    #plt.tight_layout()
    plt.legend(('ideaalimalli','data'),fontsize=20)

    plt.savefig(DATA_PATH  + 'prediction_'+dimension+'_' + method + '.png',dpi=200)
    plt.savefig('prediction_' + dimension +'.svg',format='svg')

def sorter(data,ind):
    return [data[i] for i in ind]

def get_example_IDs(ratingdata,predefined=None):
    TOPK = 10
    common = {}
    for dimension in ratingdata.keys():
        if predefined is not None:
            common[dimension] = predefined
            continue
        common[dimension]=set()
        for s in (-1, 1):
            x = np.array(list(ratingdata[dimension].keys()))
            y = np.array([ratingdata[dimension][z] for z in x])
            ind = np.argsort(s*y)
            common[dimension]=common[dimension].union(set(x[ind[0:TOPK]]))
            print('s=%i, dimension=%s, values: %s\n' % (s,dimension,str(y[ind[0:TOPK]])))
        common[dimension]=list(common[dimension])
    return common

DATA_PATH = r'D:/JanneK/Documents/git_repos/LaureaTextAnalyzer/results/best_param_run/ensemble_results/'  # data folder
RAW_DATA = r'D:\JanneK\Documents\git_repos\LaureaTextAnalyzer\results\PREPROCESSED_data.pickle'

RAW_DATA = pickle.load(open(RAW_DATA,'rb'))
filenames = glob.glob(DATA_PATH+'*.pickle')
filenames = [f for f in filenames if os.path.isfile(f)]

VISUALIZE_PREDICTIONS = ['ID004','ID123','ID177','ID344','ID353','ID060','ID318']
VISUALIZE_PREDICTIONS = get_example_IDs(RAW_DATA[4],predefined=VISUALIZE_PREDICTIONS)

result_table = pandas.DataFrame()

k_dim = 0
coef_dict = {}

weight_matrices = {}
partial_predictions ={}
handgrafted_feats={}

for file in filenames:

    resultdata = pickle.load(open(file,'rb'))

    for dimension in resultdata.keys():

        k_dim+=1

        method = get_method(resultdata[dimension]['params'])
        folds = len(list(resultdata[dimension]['train'].index)) - 1

        if dimension not in coef_dict:
            coef_dict[dimension]={}

        if method not in coef_dict[dimension]:
            coef_dict[dimension][method]=[{}]
        else:
            coef_dict[dimension][method].append({})

        y_real = resultdata[dimension]['test'].loc['total','REAL_Y']
        y_pred = resultdata[dimension]['test'].loc['total','PREDICTED_Y']
        y_null = resultdata[dimension]['test'].loc['total','NULL_Y']

        # RAW_DATA = PREPROCESSED_X, RAW_X, METADATA, CUSTOMDATA
        if 1:#dimension == 'reliability':
            if dimension not in weight_matrices:
                weight_matrices[dimension] = {}
                partial_predictions[dimension] = {}
                handgrafted_feats[dimension] = {}
            for text in VISUALIZE_PREDICTIONS[dimension]:
                ind = int(np.argwhere(np.array(resultdata[dimension]['full_data']['TEXT_IDs'])==text)[0])
                weight_mat,predictions,handgrafted = plot_predictions_advanced(
                    resultdata[dimension]['full_data']['MODEL'],
                    resultdata[dimension]['full_data']['X_DATA'][ind, :],
                    resultdata[dimension]['full_data']['FEATURES'],
                    RAW_DATA[0][text],
                    resultdata[dimension]['full_data']['PREDICTED_Y'][ind]
                )
                if text not in weight_matrices[dimension]:
                    weight_matrices[dimension][text] = []
                    partial_predictions[dimension][text]=[]
                    handgrafted_feats[dimension][text]=[]
                weight_matrices[dimension][text].append(weight_mat)
                predictions[3]=resultdata[dimension]['full_data']['REAL_Y'][ind]
                partial_predictions[dimension][text].append(predictions)
                handgrafted_feats[dimension][text].append(handgrafted)

        coef_dict[dimension][method][-1]['y_real'] = y_real
        coef_dict[dimension][method][-1]['y_pred'] = y_pred
        coef_dict[dimension][method][-1]['y_null'] = y_null

        coef_dict[dimension][method][-1]['y_pred_full'] = resultdata[dimension]['full_data']['PREDICTED_Y']

        coef_dict[dimension][method][-1]['x_ticks'] = resultdata[dimension]['test'].loc['total','TEXT_IDs']
        coef_dict[dimension][method][-1]['folds'] = resultdata[dimension]['test'].loc['total','FOLD']

        R2 = resultdata[dimension]['test'].loc['total','R2']

        ratio = resultdata[dimension]['test'].loc['total','MSE']/resultdata[dimension]['test'].loc['total','MSE_null']
        coef_dict[dimension][method][-1]['MSE_ratio'] = ratio

        ind = np.argsort(y_real)
        y_real = np.array(sorter(y_real,ind))
        y_pred = np.array(sorter(y_pred, ind))

        coefs = [[] for i in range(0,folds)]
        feats = [[] for i in range(0, folds)]
        all_feats = set()
        for fold in range(0,folds):
            try:
                coefs[fold] = resultdata[dimension]['train'].loc[fold+1, 'MODEL'].coef_
            except:
                coefs[fold] = resultdata[dimension]['train'].loc[fold + 1, 'MODEL'].best_estimator_.coef_

            mult = resultdata[dimension]['train'].loc[fold + 1, 'FEATURE_SCALE_MULTIPLIER']
            assert np.sum(mult<1e-8)==0,'bad coefficient!'
            coefs[fold] = coefs[fold]*mult
            if len(coefs[fold].shape)>1:
                coefs[fold] = coefs[fold][0,:].flatten()
            feats[fold] = resultdata[dimension]['train'].loc[fold+1, 'FEATURES']
            all_feats=all_feats.union(set(feats[fold]))
            assert len(coefs[fold])==len(feats[fold]),'Coefs and feats do not match!'

        all_coefs = np.zeros((folds+3,len(all_feats)))
        all_feats = list(all_feats)
        for k,feat in enumerate(all_feats):
            for fold in range(0, folds):
                if feat in feats[fold]:
                    all_coefs[fold,k] = coefs[fold][feats[fold].index(feat)]
            all_coefs[fold + 1, k] = np.median(all_coefs[0:folds,k])
            all_coefs[fold + 2, k] = float(not np.abs(np.sum(np.sign(all_coefs[0:folds, k])))==folds)#Utils.signrank_test(all_coefs[0:folds, k])
            all_coefs[fold + 3, k] = Utils.tvalue(all_coefs[0:folds, k])

        coef_dict[dimension][method][-1]['signcount_folds'] = np.abs(np.sum(np.sign(all_coefs[0:-3,:]),axis=0))
        coef_dict[dimension][method][-1]['median_folds'] = all_coefs[-3,:]
        coef_dict[dimension][method][-1]['mad_folds'] = robust.scale.mad(all_coefs[0:-3,:])
        coef_dict[dimension][method][-1]['label_folds'] = np.array(clean_features(all_feats))

        assert len(coef_dict[dimension][method][-1]['median_folds'])==len(coef_dict[dimension][method][-1]['label_folds'])

        try:
            coefs = resultdata[dimension]['full_data']['MODEL'].coef_
        except:
            coefs = resultdata[dimension]['full_data']['MODEL'].best_estimator_.coef_
        mult = resultdata[dimension]['full_data']['FEATURE_SCALE_MULTIPLIER']
        assert np.sum(mult < 1e-8) == 0, 'bad coefficient!'
        full_model_coefs = coefs.flatten() * mult
        coef_dict[dimension][method][-1]['median_full'] = full_model_coefs
        coef_dict[dimension][method][-1]['label_full'] = np.array(clean_features(resultdata[dimension]['full_data']['FEATURES']))

        median_coef_test = 0*full_model_coefs
        for i,label in enumerate(coef_dict[dimension][method][-1]['label_full']):
            ind = np.argwhere(coef_dict[dimension][method][-1]['label_folds']==label)
            if len(ind)>0:
                median_coef_test[i]=coef_dict[dimension][method][-1]['median_folds'][ind]
        print('... %s: correlation between FULL and FOLDED coefficient vectors: %f' % (dimension,np.corrcoef(median_coef_test,coef_dict[dimension][method][-1]['median_full'])[0,1]))

# process handgrafted features for chosen samples and save everything as .mat file (it seems easier to make plots in Matlab...)
handgrafted_coef={}
handgrafted_feat={}
for dimension in handgrafted_feats.keys():
    handgrafted_coef[dimension] = {}
    handgrafted_feat[dimension] = {}
    for text in handgrafted_feats[dimension].keys():
        handgrafted_coef[dimension][text]=[]
        commonfeat = set()
        for k,f in enumerate(handgrafted_feats[dimension][text]):
            commonfeat = commonfeat.union(set(f[1]))
        commonfeat=np.array(list(commonfeat))
        for k,f in enumerate(handgrafted_feats[dimension][text]):
            coef = {'item_weight':np.zeros(len(commonfeat)),'data_weight':np.zeros(len(commonfeat))}
            coef['item_weight'].fill(np.nan)
            coef['data_weight'].fill(np.nan)
            for kk,feat in enumerate(commonfeat):
                ind = np.argwhere(feat == f[1])
                if len(ind)>0:
                    ind=int(ind)
                    coef['item_weight'][kk]=f[0][0][ind]
                    coef['data_weight'][kk] = f[0][1][ind]
            handgrafted_coef[dimension][text].append(coef)
        handgrafted_feat[dimension][text] = commonfeat
io.savemat('text_samples_for_paper.mat',{'weight_mat':weight_matrices,'partial_predictions':partial_predictions,'handgrafted_coef':handgrafted_coef,'handgrafted_label':handgrafted_feat})

print('\n\n')
dimensions = ('reliability','sentiment','infovalue','subjectivity','textlogic','writestyle')

y_preds = {}
y_reals = {}
y_nulls = {}
mse_ratios={}
spearmancorrs= {}

print('\nEnsemble mean MSE ratios:')
for kk,dimension in enumerate(dimensions):
    y_preds[dimension]=[]
    y_reals[dimension]=[]
    y_nulls[dimension] = []

    ratios = []
    for method in coef_dict[dimension].keys():
        for k,item in enumerate(coef_dict[dimension][method]):
            ratios.append(item['MSE_ratio'])
            #if k==0:
            y_preds[dimension].append(item['y_pred'])
            y_reals[dimension].append(item['y_real'])
            y_nulls[dimension].append(item['y_null'])

    y_preds[dimension] = np.vstack(y_preds[dimension])
    y_reals[dimension] = np.vstack(y_reals[dimension])
    y_nulls[dimension] = np.vstack(y_nulls[dimension])

    mse_ratios[dimension] = {}
    mse_null = np.mean((y_nulls[dimension]-y_reals[dimension])**2)

    y_pred = np.mean(y_preds[dimension], axis=0)
    y_real = np.mean(y_reals[dimension],axis=0)
    y_null = np.mean(y_nulls[dimension],axis=0)

    mse_ratios[dimension]['single'] = np.mean((y_preds[dimension]-y_reals[dimension])**2)/mse_null
    mse_ratios[dimension]['ensemble'] = np.mean((y_pred-y_real)**2)/mse_null

    spearmancorrs[dimension] = {}
    spearmancorrs[dimension]['ensemble'] = spearmanr(y_pred,y_real).correlation

    print('%s: mean MSE ratio of INDIVIDUAL models %.3f and ENSEMBLE model %.3f (spearman corr %f)' % (dimension,mse_ratios[dimension]['single'],mse_ratios[dimension]['ensemble'],spearmancorrs[dimension]['ensemble']))

    Utils.plot_data(result_dataframe=None, filename=DATA_PATH + '%s_ENSEMBLE' % dimension, title='ENSEMBLE (test), dimension=%s, MSE ratio=%f' % (dimension,mse_ratios[dimension]['ensemble']),
                    Y_pred=y_pred,
                    Y_real=y_real,
                    Y_null=y_null,
                    X_ticks=coef_dict[dimension][method][0]['x_ticks'],
                    folds=coef_dict[dimension][method][0]['folds'])

print('\n\n')

summary_text = ''
summary_text += 'STABLE FEATURE COUNTS\ndimension, total, embedded, n-gram, custom\n'
for dimension in dimensions:

    coef_sum_ratios = []
    coef_counts = []

    for method in coef_dict[dimension].keys():
        for item in coef_dict[dimension][method]:

            good = item['signcount_folds'] > folds/2
            feats = item['label_folds'][good]
            coef_count = [0, 0, 0]
            for k in range(0, len(feats)):
                if feats[k].find('emb300') > -1:
                    kk=0
                elif feats[k].find('-g:') > -1:
                    kk=1
                else:
                    kk=2
                coef_count[kk] += 1
            assert np.sum(coef_count) == np.sum(good)

            feats = item['label_full']
            coefs = item['median_full']
            coef_sum = [0, 0, 0]
            for k in range(0, len(feats)):
                if feats[k].find('emb300') > -1:
                    kk=0
                elif feats[k].find('-g:') > -1:
                    kk=1
                else:
                    kk=2
                coef_sum[kk] += np.abs(coefs[k])

            s = np.sum(coef_sum)
            c = np.sum(coef_count)

            coef_sum_ratio = [100 * x/s for x in coef_sum]
            #coef_count_ratio = [100 * x/c for x in coef_count]

            coef_counts.append(coef_count)
            coef_sum_ratios.append(coef_sum_ratio)

    coef_counts = np.vstack(coef_counts)
    coef_sum_ratios = np.vstack(coef_sum_ratios)

    coef_sum_ratio = np.mean(coef_sum_ratios,axis=0)
    coef_count = np.mean(coef_counts,axis=0)

    print('dimension = ' + dimension)
    print(coef_counts)

    summary_text += '%s, %.1f (%.0f, %.0f, %.0f), %.1f%%, %.1f%%, %.1f%%\n' % (
        dimension,
        sum(coef_count),
        coef_count[0],
        coef_count[1],
        coef_count[2],
        coef_sum_ratio[0],
        coef_sum_ratio[1],
        coef_sum_ratio[2]
    )

print('\n\n')
print(summary_text)

common_feats = set()
for dimension in dimensions:
    for method in coef_dict[dimension].keys():
        for k,item in enumerate(coef_dict[dimension][method]):
            feats = item['label_full'][np.abs(item['median_full'])>0]
            common_feats = common_feats.union(set(feats))

common_feats = np.array(list(common_feats))
common_coefs = np.zeros((len(common_feats),6,6))
for kdim,dimension in enumerate(dimensions):
    mean_coefs = 0
    n=-1
    for method in coef_dict[dimension].keys():
        for item in coef_dict[dimension][method]:
            n+=1
            for k,feat in enumerate(common_feats):
                ind = np.argwhere(feat==item['label_full'])
                if len(ind)>1:
                    print('error')
                if len(ind)==1:
                    common_coefs[k, n, kdim] = item['median_full'][int(ind)]
common_median_coefs = np.median(common_coefs,axis=1)

print('\n')

def not_substring(t,terms):
    t = ' ' + t + ' '
    for te in terms.keys():
        if len(t)<len(te):
            if te.find(t)>-1:
                return False
    return True

term_lists=[]
term_lists_abs=[]
inds = []
for k in range(0,6):

    ind = np.argsort(-np.abs(common_median_coefs[:,k]))
    inds.append(ind)

    maxval = np.abs(common_median_coefs[ind[0],k])

    term_list = []

    coefs=[]
    print('\ndimension = %s' % dimensions[k])
    count=0
    terms = dict()
    for kk in range(0,len(common_feats)):
        coef = common_median_coefs[inds[k][kk],k]/maxval
        if common_feats[inds[k][kk]].find('emb300_')== -1:
            t = common_feats[inds[k][kk]]
            t_print=t
            j=t.find('-g: ')
            is_bow = 1
            is_lemma=0
            if t.find('LEMMA')>-1:
                is_lemma=1
            if j > -1:
                i = t.find(',type=')
                assert i>-1
                if t[-4:-1] != '+BOW':
                    is_bow = 0
                t_print = t[0:i]
                t = t[(j+7):i]
            else:
                is_lemma=-1
                if t[-6:] == '_ratio':
                    is_bow=0
                    t_print = t_print[0:-6]
                if t_print[-6:] == '_count':
                    t_print = t_print[0:-6]
            if (' ' + t + ' ') not in terms and not_substring(t,terms):
                if is_lemma==0:
                    t_print += ' (N)'
                if is_bow==1:
                    t_print += ' (C)'
                t_print=t_print.replace('#','')
                term_list.append('%i | %s | %0.2f' % (kk + 1, t_print, coef))
                coefs.append(coef)
                if count<40:
                    print('%i, %s, %0.2e' % (kk+1,t_print,coef))
                count+=1
                terms[' ' + t + ' ']=coef
    ind = np.argsort(-np.array(coefs))
    term_lists.append(np.array(term_list)[ind])
    ind = np.argsort(-np.abs(np.array(coefs)))
    term_lists_abs.append(np.array(term_list)[ind])

for k in range(0,6):
    with open(DATA_PATH + 'ENSEMBLE_ranked_feature_list_%s_noEMB.txt' % dimensions[k].upper(), 'w', encoding='utf-8') as fout:
        fout.write('dimension = %s, total non-zero features %i\n\n' % (dimensions[k],sum(np.abs(common_median_coefs[:,k])>0)) )
        for kk in range(0,len(term_lists[k])):
            fout.write('%s\n' % (term_lists[k][kk]))

SHORT_LIST_TERMS = 25
for k in range(0,6):
    with open(DATA_PATH + 'ENSEMBLE_ranked_feature_list_%s_noEMB_SHORT.txt' % dimensions[k].upper(), 'w', encoding='utf-8') as fout:
        fout.write('dimension = %s, total non-zero features %i\n\n' % (dimensions[k],sum(np.abs(common_median_coefs[:,k])>0)) )
        for kk in range(0,SHORT_LIST_TERMS):
            fout.write('%s\n' % (term_lists[k][kk]))
        fout.write('...\n')
        for kk in range(0,SHORT_LIST_TERMS):
            fout.write('%s\n' % (term_lists[k][-(SHORT_LIST_TERMS-kk)]))

SHORT_LIST_TERMS = 20
for k in range(0,6):
    with open(DATA_PATH + 'ENSEMBLE_ranked_feature_list_%s_noEMB_SHORT_onlyHandCrafted.txt' % dimensions[k].upper(), 'w', encoding='utf-8') as fout:
        fout.write('dimension = %s, total non-zero features %i\n\n' % (dimensions[k],sum(np.abs(common_median_coefs[:,k])>0)) )
        count=0
        kk=0
        while count<SHORT_LIST_TERMS and kk<len(term_lists_abs[k]):
            if term_lists_abs[k][kk].find('g: ')==-1 and term_lists_abs[k][kk].find('_tag')==-1:
                fout.write('%s\n' % (term_lists_abs[k][kk]))
                count+=1
            kk += 1

from scipy.stats import spearmanr,pearsonr
def get_corrmat(data):
    n=data.shape[1]
    corvals1 = np.zeros((n, n))
    corvals2 = np.zeros((n, n))
    for k1 in range(0, n):
        for k2 in range(0, n):
            rho, pval = spearmanr(data[:, k1], data[:, k2])
            corvals1[k1, k2] = rho
            rho, pval = pearsonr(data[:, k1], data[:, k2])
            corvals2[k1, k2] = rho
    return corvals1,corvals2


##########################

if False:
    cormat_spearman = np.zeros(6)
    cormat_pearson = np.zeros(6)
    print('\nPearson correlation matrix (diagonal):')
    for kdim, dimension in enumerate(dimensions):
        count = 0
        for k in list(itertools.product([0, 1], repeat=3)):
            common_feats = set()
            for kmet,method in enumerate(('SVM', 'ElasticNet', 'Ridge')):
                item = coef_dict[dimension][method][k[kmet]]
                good = item['signcount_folds'] > 5
                feats = item['label_folds'][good]
                n=len(feats)
                feats = [x for x in feats if x in item['label_full']]
                missing_count = n - len(feats)
                common_feats = common_feats.union(set(feats))
            common_feats = np.array(list(common_feats))
            common_coefs = np.zeros((len(common_feats),3))
            for kmet, method in enumerate(('SVM', 'ElasticNet', 'Ridge')):
                item = coef_dict[dimension][method][k[kmet]]
                for kk,feat in enumerate(common_feats):
                    ind = np.argwhere(feat==item['label_full'])
                    if len(ind)>1:
                        print('error')
                    if len(ind)==1:
                        common_coefs[kk,kmet] = item['median_full'][int(ind)]
            c1,c2 = get_corrmat(common_coefs)
            cormat_spearman[kdim]+=np.mean(c1[np.triu_indices(c1.shape[0], k = 1)])
            cormat_pearson[kdim]+=np.mean(c2[np.triu_indices(c2.shape[0], k = 1)])
            count += 1
        cormat_spearman[kdim] = cormat_spearman[kdim]/count
        cormat_pearson[kdim] = cormat_pearson[kdim]/count
        print('... %s: %s' % (dimension,str(cormat_pearson[kdim])))

##########################

if True:
    cormat_spearman = 0
    cormat_pearson = 0
    count=0
    for method in ('SVM', 'ElasticNet', 'Ridge'):
        for k in list(itertools.product([0, 1], repeat=6)):
            common_feats = set()
            for kdim, dimension in enumerate(dimensions):
                item = coef_dict[dimension][method][k[kdim]]
                good = item['signcount_folds'] > 5
                feats = item['label_folds'][good]
                n=len(feats)
                feats = [x for x in feats if x in item['label_full']]
                missing_count = n - len(feats)
                common_feats = common_feats.union(set(feats))
            common_feats = np.array(list(common_feats))
            common_coefs = np.zeros((len(common_feats),6))
            for kdim, dimension in enumerate(dimensions):
                item = coef_dict[dimension][method][k[kdim]]
                for kk,feat in enumerate(common_feats):
                    ind = np.argwhere(feat==item['label_full'])
                    if len(ind)>1:
                        print('error')
                    if len(ind)==1:
                        common_coefs[kk,kdim] = item['median_full'][int(ind)]
            c1,c2 = get_corrmat(common_coefs)
            cormat_spearman+=c1
            cormat_pearson+=c2
            count += 1

    cormat_spearman = cormat_spearman/count
    cormat_pearson = cormat_pearson/count

    print('Pearson correlation matrix (off-diagonal):\n%s\n',str(cormat_pearson))

#############################

plt.close('all')
fig = plt.figure(1)
DPI = float(fig.dpi)
fig.set_size_inches(1300/DPI,750/DPI)

fig1 = plt.figure(2)
fig1.set_size_inches(1000/DPI,750/DPI)

common_ylim = [0,0]

mydata = {}

#x_ticks = np.array((-0.015,-0.01,-0.005,0,0.005,0.010,0.015))
for kk,dimension in enumerate([('reliability','Trustworthiness'),('infovalue','Information'),('sentiment','Sentiment'),('writestyle','Clarity'),('textlogic','Logic'),('subjectivity','Neutrality')]):

    titlestr = dimension[1]
    dimension = dimension[0]

    common_feats = set()
    for method in coef_dict[dimension].keys():
        for k, item in enumerate(coef_dict[dimension][method]):
            feats = item['label_folds'][item['signcount_folds'] > 5]
            common_feats = common_feats.union(set(feats))

    common_feats = np.array(list(common_feats))
    common_coefs = np.zeros((len(common_feats),3*2))
    n=-1
    for method in coef_dict[dimension].keys():
        for item in coef_dict[dimension][method]:
            n+=1
            for k, feat in enumerate(common_feats):
                ind = np.argwhere(feat == item['label_full'])
                if len(ind)==0:
                    common_coefs[k, n] = 0
                if len(ind) > 1:
                    print('error')
                if len(ind) == 1:
                    common_coefs[k, n] = item['median_full'][int(ind)]
    common_median_coefs = np.median(common_coefs, axis=1)
    good = np.abs(common_median_coefs)>0
    common_median_coefs=common_median_coefs[good]
    common_feats = common_feats[good]

    ind = np.argsort(common_median_coefs)
    common_median_coefs = common_median_coefs[ind]
    common_feats = common_feats[ind]

    ind_emd = [i[0] for i in enumerate(common_feats) if i[1].find('emb300_')>-1]
    ind_ngram = [i[0] for i in enumerate(common_feats) if i[1].find('-g: ')>-1]
    ind_custom = [i[0] for i in enumerate(common_feats) if i[0] not in ind_emd and i[0] not in ind_ngram]

    mydata[dimension] = {'common_feats':common_feats,'common_median_coefs':common_median_coefs,'ind_emd':np.array(ind_emd)+1,'ind_ngram':np.array(ind_ngram)+1,'ind_custom':np.array(ind_custom)+1}

    assert len(ind_emd)+len(ind_ngram)+len(ind_custom)==len(ind)

    plt.figure(fig.number)
    plt.subplot(2, 3, kk + 1)
    col = ['red', 'black', 'green']
    colors=[None for i in range(0,len(ind))]
    for k,indset in enumerate([ind_emd,ind_ngram,ind_custom]):
        for i in indset:
            colors[i] = col[k]
    y = range(0,len(ind))
    plt.barh(y,common_median_coefs,1,color=colors)
    ax = fig.get_axes()[kk]
    if kk>2:
        ax.set_xlabel('Coefficient (sorted)',fontsize=18)
    if kk==0 or kk==3:
        ax.set_ylabel('Weight',fontsize=18)
    #ax.set_title('%s' % titlestr,fontsize=18)
    #x_ticks, x_ticks_labels = plt.xticks()
    #ax.set_xticklabels(x_ticks,rotation=-90,ha='center',fontsize=16)
    plt.plot([0,0],[-1,len(ind)],color='black')
    ax.set_ylim([-6,len(ind)+7])
    xxx = 0.01*(np.max(common_median_coefs) - np.min(common_median_coefs))
    xxx = [np.min(common_median_coefs) - xxx, np.max(common_median_coefs) + xxx]
    ax.set_xlim(xxx)

    print('--- x-limits for %s: %f - %f' % (dimension,xxx[0],xxx[1]))

    plt.xticks(rotation=-45,fontsize=16,ha='center')
    plt.yticks(fontsize=16)
    #ax.tick_params(axis='x', which='both', labelsize=6)
    #ax.tick_params(axis='y',labelsize=19,)
    plt.show(block=False)

    plt.figure(fig1.number)
    col = ['red','black','green']
    siz = [7,0,8]
    mark = ['o','o','s']
    colors=[None for i in range(0,len(ind))]
    sizes = [None for i in range(0, len(ind))]
    markers = [None for i in range(0, len(ind))]
    for k,indset in enumerate([ind_emd,ind_ngram,ind_custom]):
        for i in indset:
            colors[i] = col[k]
            sizes[i] = siz[k]
            markers[i] = mark[k]
    x = np.arange(0,len(ind))
    x = x / (len(x) - 1)
    y = common_median_coefs/np.sum(np.abs(common_median_coefs))
    for j in range(len(x)):
        #ax.scatter([xp], [yp], marker=m)
        plt.scatter(x[j],y[j],s=sizes[j],c=colors[j],marker=markers[j])
    ax = fig1.get_axes()[0]
    ax.set_xlabel('Coefficient number (normalized)',fontsize=18)
    ax.set_ylabel('Weight (normalized)',fontsize=18)
    #ax.set_title('%s' % titlestr,fontsize=18)
    #x_ticks, x_ticks_labels = plt.xticks()
    #ax.set_xticklabels(x_ticks,rotation=-90,ha='center',fontsize=16)
    plt.plot([0,1],[0,0],color='black')
    ax.set_xlim([-0.01,1.01])

    common_ylim = [np.minimum(common_ylim[0],np.min(y)),np.maximum(common_ylim[1],np.max(y))]

    ax.set_ylim(common_ylim)

    print('--- x-limits for %s: %f - %f' % (dimension,xxx[0],xxx[1]))

    plt.xticks(rotation=-45,fontsize=16,ha='center')
    plt.yticks(fontsize=16)
    #ax.tick_params(axis='x', which='both', labelsize=6)
    #ax.tick_params(axis='y',labelsize=19,)
    plt.show(block=False)

io.savemat('common_median_hist_data_for_paper.mat',mydata)

plt.figure(fig.number)
plt.subplots_adjust(left  = 0.09,  # the left side of the subplots of the figure
                    right = 0.85,    # the right side of the subplots of the figure
                    bottom = 0.17,   # the bottom of the subplots of the figure
                    top = 0.95,      # the top of the subplots of the figure
                    hspace = 0.32,   # the amount of height reserved for blank space between subplots
                    wspace = 0.26)   # the amount of width reserved for white space between subplots

from matplotlib.lines import Line2D

labels = ('embedded', 'n-grams', 'hand-crafted')
legend_elements = [Line2D([0], [0], color=col[0], lw=4,label=labels[0]),
                Line2D([0], [0], color=col[1], lw=4,label=labels[1]),
                Line2D([0], [0], color=col[2], lw=4,label=labels[2])]

ax.legend(handles=legend_elements, loc='center', bbox_to_anchor=(1.05, 0.5), fontsize=16)

plt.savefig(DATA_PATH+'rating_histogram_comparison.pdf',dpi=200)
plt.savefig(DATA_PATH+'rating_histogram_comparison.png')



plt.figure(fig1.number)
plt.subplots_adjust(left  = 0.09,  # the left side of the subplots of the figure
                    right = 0.85,    # the right side of the subplots of the figure
                    bottom = 0.17,   # the bottom of the subplots of the figure
                    top = 0.95,      # the top of the subplots of the figure
                    hspace = 0.32,   # the amount of height reserved for blank space between subplots
                    wspace = 0.26)   # the amount of width reserved for white space between subplots

from matplotlib.lines import Line2D

labels = ('embedded', 'n-grams', 'hand-crafted')
legend_elements = [Line2D([0], [0], color=col[0], lw=4,label=labels[0]),
                Line2D([0], [0], color=col[1], lw=4,label=labels[1]),
                Line2D([0], [0], color=col[2], lw=4,label=labels[2])]

ax.legend(handles=legend_elements, loc='center', bbox_to_anchor=(1.05, 0.5), fontsize=16)

plt.savefig(DATA_PATH+'rating_histogram_comparison_scaled.pdf',dpi=200)
plt.savefig(DATA_PATH+'rating_histogram_comparison_scaled.png')