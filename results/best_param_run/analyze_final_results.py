# plot results computed by COMPARISON.py
# code is ugly, but works...

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas
import glob
import os
import Utils
from statsmodels import robust

def clean_features(arr):
    for k,s in enumerate(arr):
        s = s.replace('END_OF_PARAGRAPH', 'EOP')
        if s.find('term=') > -1:
            s = s.replace('term=', '%i-gram: ' % (s.count(' ') + 1))
        arr[k]=s
    return arr

if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    DATA_PATH = r'D:/JanneK/Documents/git_repos/LaureaTextAnalyzer/results/best_param_run/best_SVM/'  # data folder

    filenames = glob.glob(DATA_PATH+'*.pickle')
    filenames = [f for f in filenames if os.path.isfile(f)]

    result_table = pandas.DataFrame()

    def sorter(data,ind):
        return [data[i] for i in ind]

    k_dim = 0
    coef_table_info = pandas.DataFrame()
    coef_dict = {}

    for file in filenames:

        resultdata = pickle.load(open(file,'rb'))

        for dimension in resultdata.keys():

            coef_dict[dimension]={}

            k_dim+=1

            y_real = resultdata[dimension]['test'].loc['total','REAL_Y']
            y_pred = resultdata[dimension]['test'].loc['total','PREDICTED_Y']

            R2 = resultdata[dimension]['test'].loc['total','R2']

            ind = np.argsort(y_real)
            y_real = np.array(sorter(y_real,ind))
            y_pred = np.array(sorter(y_pred, ind))

            #width = 0.1  # the width of the bars

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

            #autolabel(rects1)
            #autolabel(rects2)
            #autolabel(rects3)
            #autolabel(rects4)

            #plt.tight_layout()
            plt.legend(('ideaalimalli','data'),fontsize=20)

            plt.savefig(DATA_PATH  + 'prediction_'+dimension+'.png',dpi=200)
            #plt.savefig('prediction_' + dimension +'.svg',format='svg')

            if 1:

                folds = len(list(resultdata[dimension]['train'].index))-1
                coefs = [[] for i in range(0,folds)]
                feats = [[] for i in range(0, folds)]
                all_feats = set()
                for fold in range(0,folds):
                    try:
                        coefs[fold] = resultdata[dimension]['train'].loc[fold+1, 'MODEL'].coef_
                    except:
                        coefs[fold] = resultdata[dimension]['train'].loc[fold + 1, 'MODEL'].best_estimator_.coef_
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
                    all_coefs[fold + 2, k] = float(not np.abs(np.sum(np.sign(all_coefs[0:folds, k])))==10)#Utils.signrank_test(all_coefs[0:folds, k])
                    all_coefs[fold + 3, k] = Utils.tvalue(all_coefs[0:folds, k])

                ind = np.argsort(all_coefs[-2,:])
                all_coefs = all_coefs[:, ind]
                all_feats = [all_feats[i].replace('_ ','_0') for i in ind]

                with open(DATA_PATH+'feature_list_%s.txt' % dimension.upper(),'w',encoding='utf-8') as fout:
                    fout.write('dimension = %s\nparams: %s\n\n' % (dimension,resultdata[dimension]['params']))
                    fout.write('total %i terms (present in one or more folds) with %i embeddings\n' % (len(all_feats),sum([1 for i in all_feats if i[0:3]=='emb'])))
                    fout.write('term\tpval\tmedian\tcoefficients\n')
                    coef_sum = [0,0,0]
                    coef_count = [0,0,0]
                    for k in range(0,len(all_feats)):
                        a = np.array_str(all_coefs[0:folds,k])
                        a=a.replace('\n','')
                        fout.write('%s\t%f\t%f\t%s\n' % (all_feats[k],all_coefs[-2,k],all_coefs[-3,k],a))
                        is_significant=False
                        if all_coefs[-2,k]<0.01:
                            is_significant = True
                            if all_feats[k].find('emb300')>-1:
                                coef_sum[0]+=np.abs(all_coefs[folds,k])
                                coef_count[0] += 1
                            elif all_feats[k].find('term=')>-1:
                                coef_sum[1]+=np.abs(all_coefs[folds,k])
                                coef_count[1] += 1
                            else:
                                coef_sum[2]+=np.abs(all_coefs[folds,k])
                                coef_count[2] += 1

                coef_sum_ratio = [100*x/sum(coef_sum) for x in coef_sum]
                coef_count_ratio = [100*x/sum(coef_count) for x in coef_count]
                if k_dim==1:
                    print('feature counts at p<0.01\ndimension\ttotal\tembedded\tn-gram\tcustom')
                print('%s\t%i\t%.0f%% (%.0f)\t%.0f%% (%.0f)\t%.0f%% (%.0f)' % (
                    dimension,
                    sum(coef_count),
                    coef_sum_ratio[0],
                    coef_count[0],
                    coef_sum[1],
                    coef_count[1],
                    coef_sum_ratio[2],
                    coef_count[2]
                      ))

                coef_table_info.loc[dimension,'total'] = sum(coef_count)
                coef_table_info.loc[dimension, 'embedded'] = coef_count[0]
                coef_table_info.loc[dimension, 'n-gram'] = coef_count[1]
                coef_table_info.loc[dimension, 'custom'] = coef_count[2]
                coef_table_info.loc[dimension, 'embedded (coef)'] = coef_sum_ratio[0]
                coef_table_info.loc[dimension, 'n-gram (coef)'] = coef_sum_ratio[1]
                coef_table_info.loc[dimension, 'custom (coef)'] = coef_sum_ratio[2]

                ind = np.argsort(-np.abs(all_coefs[-1,:]))
                all_coefs = all_coefs[:, ind]
                all_feats = [all_feats[i] for i in ind]
                with open(DATA_PATH+'ranked_feature_list_%s.txt' % dimension.upper(), 'w', encoding='utf-8') as fout:
                    fout.write('dimension = %s, file = %s\n\n' % (dimension,file))
                    fout.write('term\tpval\tmedian\tt-value\n------------------------------------\n')
                    terms = set()
                    good=0
                    row=0
                    while good<100:
                        if all_feats[row].find('emb300')==-1:
                            s = all_feats[row]
                            fout.write('%s\t%f\t%f\t%f\n' % (s, all_coefs[-2, row], all_coefs[-3, row],all_coefs[-1, row]))
                            good+=1
                        row+=1

                with open(DATA_PATH+'ranked_feature_list_%s_CLEANED.txt' % dimension.upper(), 'w', encoding='utf-8') as fout:
                    fout.write('dimension = %s, file = %s\n\n' % (dimension,file))
                    fout.write('term\tt-value\n--------------------------------\n')
                    terms = set()
                    good=0
                    row=0
                    while good<100:
                        if all_feats[row].find('emb300')==-1:
                            s = all_feats[row]
                            i = s.find(',type=')
                            is_absolute = 0
                            if i>-1:
                                if s.find('BOW')>-1:
                                    is_absolute=1
                                s = s[0:i]
                            raw_term = s
                            is_ngram=0
                            if s.find('term=')>-1:
                                is_ngram = 1
                                raw_term = s[5:]
                                s=s.replace('term=','%i-gram: ' % (s.count(' ')+1))
                            s=s.replace('END_OF_PARAGRAPH','EOP')
                            if raw_term not in terms:
                                terms.add(raw_term)
                                if is_absolute==1:
                                    s+=' (A)'
                                fout.write('%s\t%f\n' % (s,all_coefs[-1, row]))
                                good+=1
                        row+=1
                with open(DATA_PATH+'ranked_feature_list_%s_CUSTOMFEAT.txt' % dimension.upper(), 'w', encoding='utf-8') as fout:
                    fout.write('dimension = %s, file = %s\n\n' % (dimension,file))
                    fout.write('custom_feature\tt-value\n--------------------------------\n')
                    terms = set()
                    good=0
                    row=0
                    while good<10 and row<len(all_feats):
                        #print('row=%i, all_coefs.shape=%s, len(all_feats)=%s' % (row,str(all_coefs.shape),str(len(all_feats))))
                        if all_feats[row].find('emb300')==-1 and all_feats[row].find('term=')==-1 and not np.isnan(all_coefs[-1, row]):
                            s = all_feats[row]
                            i = s.find(',type=')
                            is_absolute = 0
                            if i>-1:
                                if s.find('BOW')>-1:
                                    is_absolute=1
                                s = s[0:i]
                            raw_term = s
                            is_ngram=0
                            if s.find('term=')>-1:
                                is_ngram = 1
                                raw_term = s[5:]
                                s=s.replace('term=','%i-gram: ' % (s.count(' ')+1))
                            s=s.replace('END_OF_PARAGRAPH','EOP')
                            if raw_term not in terms:
                                terms.add(raw_term)
                                if is_absolute==1:
                                    s+=' (A)'
                                fout.write('%s\t%f\n' % (s,all_coefs[-1, row]))
                                good+=1
                        row+=1

                coef_dict[dimension]['signcount'] = np.abs(np.sum(np.sign(all_coefs[0:-3,:]),axis=0))
                coef_dict[dimension]['median'] = all_coefs[-3,:]
                coef_dict[dimension]['mad'] = robust.scale.mad(all_coefs[0:-3,:])
                coef_dict[dimension]['label'] = np.array(clean_features(all_feats))

                assert len(coef_dict[dimension]['median'])==len(coef_dict[dimension]['label'])

    print('\n\n')

    for dimension in list(coef_table_info.index):
        print('%s\t%i\t%.0f%% (%.0f)\t%.0f%% (%.0f)\t%.0f%% (%.0f)' % (dimension,
                                                                       coef_table_info.loc[dimension,'total'],
                                                                       coef_table_info.loc[dimension, 'embedded (coef)'],coef_table_info.loc[dimension,'embedded'],
                                                                       coef_table_info.loc[dimension, 'n-gram (coef)'],coef_table_info.loc[dimension, 'n-gram'],
                                                                       coef_table_info.loc[dimension, 'custom (coef)'],coef_table_info.loc[dimension, 'custom'], ))
    print('%s\t%i\t%.0f%% (%.0f)\t%.0f%% (%.0f)\t%.0f%% (%.0f)' % ('mean',
                                                                   coef_table_info['total'].mean(0),
                                                                   coef_table_info['embedded (coef)'].mean(0),coef_table_info['embedded'].mean(0),
                                                                   coef_table_info['n-gram (coef)'].mean(0),coef_table_info['n-gram'].mean(0),
                                                                   coef_table_info['custom (coef)'].mean(0),coef_table_info['custom'].mean(0)))

    common_feats = set()
    dimensions = list(coef_dict.keys())
    for dimension in dimensions:
        good = coef_dict[dimension]['signcount'] == 10
        common_feats = common_feats.union(set(coef_dict[dimension]['label'][good]))
    common_feats = list(common_feats)
    common_coefs=np.zeros((len(common_feats),6))
    for k1,dimension in enumerate(dimensions):
        for k, feat in enumerate(common_feats):
            if feat in coef_dict[dimension]['label']:
                common_coefs[k, k1] = coef_dict[dimension]['median'][list(coef_dict[dimension]['label']).index(feat)]

    from scipy.stats import spearmanr
    corvals=np.zeros((6,6))
    for k1 in range(0,6):
        for k2 in range(0,6):
            rho, pval = spearmanr(common_coefs[:, k1],common_coefs[:, k2])
            corvals[k1,k2]=rho

    plt.close('all')
    fig = plt.figure(1)
    DPI = float(fig.dpi)
    fig.set_size_inches(1300/DPI,750/DPI)

    #x_ticks = np.array((-0.015,-0.01,-0.005,0,0.005,0.010,0.015))
    for kk,dimension in enumerate([('reliability','Trustworthiness'),('infovalue','Info'),('sentiment','Sentiment')]):
        plt.subplot(1,3,kk+1)
        titlestr = dimension[1]
        dimension = dimension[0]
        good=coef_dict[dimension]['signcount']==10
        coef_dict[dimension]['median'] = coef_dict[dimension]['median'][good]
        coef_dict[dimension]['label']=coef_dict[dimension]['label'][good]
        coef_dict[dimension]['mad']=coef_dict[dimension]['mad'][good]

        ind = np.argsort(coef_dict[dimension]['median'])
        coef_dict[dimension]['median'] = coef_dict[dimension]['median'][ind]
        coef_dict[dimension]['mad'] = coef_dict[dimension]['mad'][ind]

        ind_emd = [i[0] for i in enumerate(coef_dict[dimension]['label']) if i[1].find('emb300')>-1]
        ind_ngram = [i[0] for i in enumerate(coef_dict[dimension]['label']) if i[1].find('-gram')>-1]
        ind_custom = [i[0] for i in enumerate(coef_dict[dimension]['label']) if i[0] not in ind_emd and i[0] not in ind_ngram]

        assert len(ind_emd)+len(ind_ngram)+len(ind_custom)==len(ind)

        col = ['green','blue','red']
        colors=[None for i in range(0,len(ind))]
        for k,indset in enumerate([ind_emd,ind_ngram,ind_custom]):
            for i in indset:
                colors[i] = col[k]

        y = range(0,len(ind))
        plt.barh(y,coef_dict[dimension]['median'],1,color=colors)
        ax = fig.get_axes()[kk]
        ax.set_xlabel('Coefficient',fontsize=18)
        if kk==0:
            ax.set_ylabel('Feature (sorted)',fontsize=18)
        ax.set_title('%s' % titlestr,fontsize=18)
        #x_ticks, x_ticks_labels = plt.xticks()
        #ax.set_xticklabels(x_ticks,rotation=-90,ha='center',fontsize=16)
        plt.plot([0,0],[-1,len(ind)],color='black')
        ax.set_ylim([-3,len(ind)+2])
        plt.xticks(rotation=-45,fontsize=16,ha='center')
        plt.yticks(fontsize=16)
        #ax.tick_params(axis='x', which='both', labelsize=6)
        #ax.tick_params(axis='y',labelsize=19,)

    plt.subplots_adjust(left  = 0.09,  # the left side of the subplots of the figure
                        right = 0.80,    # the right side of the subplots of the figure
                        bottom = 0.20,   # the bottom of the subplots of the figure
                        top = 0.92,      # the top of the subplots of the figure
                        hspace = 0.0,   # the amount of height reserved for blank space between subplots
                        wspace = 0.25)   # the amount of width reserved for white space between subplots
    plt.show(block=False)

    from matplotlib.lines import Line2D

    labels = ('embedded', 'n-grams', 'hand-crafted')
    legend_elements = [Line2D([0], [0], color=col[0], lw=4,label=labels[0]),
                    Line2D([0], [0], color=col[1], lw=4,label=labels[1]),
                    Line2D([0], [0], color=col[2], lw=4,label=labels[2])]

    ax.legend(handles=legend_elements, loc='center', bbox_to_anchor=(1.05, 0.5), fontsize=16)

    plt.savefig(DATA_PATH+'coefficient_plot.pdf',dpi=200)
    plt.savefig(DATA_PATH+'rating_histogram_comparison.png')