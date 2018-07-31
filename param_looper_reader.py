# loop over parameter candidates and keep track of best test results (smallest MSE)
import pickle
from sortedcontainers import SortedListWithKey
import os
import pandas
from tabulate import tabulate

#RESULT_FOLDERS = ('D:/JanneK/Documents/git_repos/LaureaTextAnalyzer/results/cluster_results/cluster_results_EN_new/',)

ROOT_FOLDER = r'D:\JanneK\Documents\git_repos\LaureaTextAnalyzer\results\cluster_results'

RESULT_FOLDERS = [x[0]+os.sep for x in os.walk(ROOT_FOLDER) if x[0] is not ROOT_FOLDER]
#RESULT_FOLDERS = [r'D:\JanneK\Documents\git_repos\LaureaTextAnalyzer\results\cluster_results\runs_part1\cluster_results_MLP'+os.sep]

#RESULT_FOLDERS = (r'D:/JanneK/Documents/git_repos/LaureaTextAnalyzer/results/cluster_results/cluster_results_MLP_final_addition/',)

PRINT_TOP = 80

all_results = {}
N_sets = 0

def get_method(string):
    s='\'Algorithm\': (\''
    ind1 = string.find(s)
    assert ind1>-1
    ind1+=len(s)
    ind2 = string[ind1:].find('\'')
    return string[ind1:(ind1+ind2)]

for RESULT_FOLDER in RESULT_FOLDERS:

    filelist = [RESULT_FOLDER + f for f in os.listdir(RESULT_FOLDER) if f.endswith("RESULTS.pickle")]
    if len(filelist)<5:
        continue

    print('Total %i result files found in folder %s' % (len(filelist),RESULT_FOLDER))

    for batch,file in enumerate(filelist):

        try:
            result_arr = pickle.load(open(file,'rb'))
        except:
            print('failed to open file %s' % file)
            raise(Exception(''))

        print('... adding batch %i (with %i sets)' % (batch + 1,len(result_arr)))

        N_sets+=len(result_arr)

        #params_arr,_ = pickle.load(open(file[0:-15] + '.pickle', 'rb'))

        if batch==0:
            dimensions = list(result_arr[0].keys())
            fold_count = len(result_arr[0][dimensions[0]]['train']['R'])-1

        for kk,dimension in enumerate(dimensions):
            if dimension not in all_results:
                all_results[dimension] = []
            for k in range(len(result_arr)):
                all_results[dimension].append(result_arr[k][dimension])

                # if kk==0:
                #     s2 = result_arr[k][dimension]['params'][1:-1]
                #     for key in params_arr[k]['LearnParams']:
                #         s = str(params_arr[k]['LearnParams'][key])
                #         if s2.find(s)<0:
                #             raise (Exception('!!!!! Parameters not identical !!!!!'))

print('Total %i parameter sets added' % N_sets)

top_results={}

for dimension in all_results.keys():
    top_results[dimension]={}
    for k in range(len(all_results[dimension])):
        ratio = all_results[dimension][k]['test']['MSE']['total'] / all_results[dimension][k]['test']['MSE_null']['total']
        method = get_method(all_results[dimension][k]['params'])
        item = {'ratio':ratio,'params':all_results[dimension][k]['params'],'method':method,'all_data':all_results[dimension][k]}
        if method not in top_results[dimension]:
            top_results[dimension][method] = SortedListWithKey(key=lambda x: x['ratio'])
        top_results[dimension][method].add(item)

result_table = pandas.DataFrame()

for dimension in top_results.keys():
    for method in top_results[dimension].keys():
        with open(ROOT_FOLDER+os.sep+str(dimension)+'_' + str(method) + '_POOLED_result_summary.txt','w',encoding='utf-8') as fout:
            fout.write('TOP %i results for \'%s\' using method %s\n\n' % (PRINT_TOP,dimension,method))
            for k in range(PRINT_TOP):

                fout.write('______ RANK = %i ______\n' % (k + 1))

                params = top_results[dimension][method][k]

                fout.write('TEST: R = %f, R2 = %f, MSE = %f (null %f, ratio %f), f1 = %f (null %f)\n' % (
                    params['all_data']['test']['R']['total'],
                    params['all_data']['test']['R2']['total'],
                    params['all_data']['test']['MSE']['total'],
                    params['all_data']['test']['MSE_null']['total'],
                    params['ratio'],
                    params['all_data']['test']['F1']['total'],
                    params['all_data']['test']['F1_null']['total']))
                fout.write('TRAIN: R = %f, R2 = %f, MSE = %f (null %f, ratio %f), f1 = %f (null %f)\n' % (
                    params['all_data']['train']['R']['total'],
                    params['all_data']['train']['R2']['total'],
                    params['all_data']['train']['MSE']['total'],
                    params['all_data']['train']['MSE_null']['total'],
                    params['all_data']['train']['MSE']['total'] / params['all_data']['train']['MSE_null']['total'],
                    params['all_data']['train']['F1']['total'],
                    params['all_data']['train']['F1_null']['total']))
                fout.write('Learn parameters: ' + params['params'] + '\n')

                if k==0:
                    result_table.loc[method, dimension] = params['ratio']

print('All done!')

print(result_table)
result_table.to_csv('top_models_table_formatted_new.csv')

print('\n')

print(tabulate(result_table, headers='keys',floatfmt=".3f"))