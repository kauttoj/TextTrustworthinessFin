# loop over parameter candidates and keep track of best test results (smallest MSE)

import pickle
import os
import Utils
import copy
import sys
import random
#import hashlib
#import numpy as np

SETS_PER_JOB = 3  # how many parameters sets to execute per job
IS_CLUSTER = 2  # 0=no cluster, 1=simulate cluster, 2=cluster
ALLOW_FAILURE = 1
MAX_JOBS = 800

assert IS_CLUSTER in [0,1,2],'IS_CLUSTER must be 1, 2 or 3'

#RootPath = r'D:/JanneK/Documents/git_repos/LaureaTextAnalyzer/'
#ResultPath = r'D:/JanneK/Documents/git_repos/LaureaTextAnalyzer/results/looper_best_params/'
RootPath = r'/home/kauttoj2/CODES/LaureaTextAnalyzer/'
ResultPath = r'/home/kauttoj2/CODES/LaureaTextAnalyzer/results/cluster_results_RIDGE_final_second/'

if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

    Utils.isfolder(ResultPath)

    params_file = ResultPath + 'all_paramsets.pickle'
    params_file_batch = ResultPath + 'all_paramsets_batch.pickle'

    if os.path.isfile(params_file) and IS_CLUSTER>0:
        print('Found old parameters, using them! Delete all files in folder %s to restart computing clean.' % ResultPath)
        paramsets = pickle.load(open(params_file, 'rb'))
    else:

        # NOTE: all entries are lists of lists!
        #
     #   params['TextTypes'] = (('LEMMA',),('LEMMA','POS',),('LEMMA','NORMAL',))#('NORMAL', 'LEMMA', 'POS',),('LEMMA', 'POS',),)
     #   params['FeatureMethod'] = (('BOW_2','TFIDF_2','EMBEDDING','CUSTOM','TAGS',),('TFIDF_2','EMBEDDING','CUSTOM','TAGS',),('TFIDF_2','EMBEDDING','TAGS',))#[[]]#('BOW_2', 'TFIDF_2','EMBEDDING','CUSTOM','TAGS'),('TFIDF_2', 'EMBEDDING', 'CUSTOM', 'TAGS'),('TFIDF_3', 'EMBEDDING', 'CUSTOM', 'TAGS'))
     #   params['FeatureSelection'] = (('regression','all',(50,100,250,500,)),('fisher','all',(50,100,250,500,)),('regression','global',(100,250,500,800,1000)))
        FeatureSelection=((None,),
                          ('regression', 'global', 800),
                          ('regression', 'global', 1100),
                          ('regression', 'global',1500),
                          ('fisher', 'global',800),
                          ('fisher', 'global', 1500),
                          )
        FeatureMethod = (
                         ('TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'),
                         ('BOW_3','TFIDF_3','EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'),
                         )
        TextTypes = (('LEMMA', 'NORMAL'), ('LEMMA',), ('LEMMA', 'POS'),)
        pass1_features = ((200,),(400,),(600,),(800,),(2000,))
        FeatureScaler = (('MaxAbsScaler',),('StandardScaler',))
        pass1_features_SVD = ((80,),(None,),)

        LearnParams = (
            # {'FeatureScaler': FeatureScaler,
            #  'pass1_features': pass1_features,
            #  'FeatureSelection': FeatureSelection,
            #  'TextTypes': TextTypes,
            #  'pass1_features_SVD' : pass1_features_SVD,
            #  'FeatureMethod': FeatureMethod,
            #  'Algorithm': ('ElasticNet', {'l1_ratio': (0.01,0.10,)})},
            # {'FeatureScaler': FeatureScaler,
            #  'pass1_features': pass1_features,
            #  'FeatureSelection': FeatureSelection,
            #  'TextTypes': TextTypes,
            #  'pass1_features_SVD': pass1_features_SVD,
            #  'FeatureMethod': FeatureMethod,
            #  'Algorithm': ('ElasticNet_multitarget', {'l1_ratio': (0.01, 0.10,)})},
            # {'FeatureScaler': FeatureScaler,
            #  'pass1_features': pass1_features,
            #  'FeatureSelection': FeatureSelection,
            #  'TextTypes': TextTypes,
            #  'pass1_features_SVD': pass1_features_SVD,
            #  'FeatureMethod': FeatureMethod,
            #  'Algorithm':('MLP', {'hidden_layer_sizes': (5,10,20,25)})},
            # {'FeatureScaler': FeatureScaler,
            #  'pass1_features': pass1_features,
            #  'FeatureSelection': FeatureSelection,
            #  'TextTypes': TextTypes,
            #  'pass1_features_SVD': pass1_features_SVD,
            #  'FeatureMethod': FeatureMethod,
            #  'Algorithm':('MLP_multitarget', {'hidden_layer_sizes': (5,10,20,25)})},
            # {'FeatureScaler': FeatureScaler,
            #  'pass1_features': pass1_features,
            #  'FeatureSelection': FeatureSelection,
            #  'TextTypes': TextTypes,
            #  'pass1_features_SVD' : pass1_features_SVD,
            #  'FeatureMethod': FeatureMethod,
            #  'Algorithm': ('RandomForest', {'n_estimators': (30,50,100,),'max_depth':(2,3,4)})},
            # {'FeatureScaler': FeatureScaler,
            #  'pass1_features': pass1_features,
            #  'FeatureSelection': FeatureSelection,
            #  'TextTypes': TextTypes,
            #  'pass1_features_SVD' : pass1_features_SVD,
            #  'FeatureMethod': FeatureMethod,
            #  'Algorithm': ('RandomForest_multitarget', {'n_estimators': (30,50,100,),'max_depth':(2,3,4)})},
            {'FeatureScaler': FeatureScaler,
             'pass1_features': pass1_features,
             'FeatureSelection': FeatureSelection,
             'TextTypes': TextTypes,
             'pass1_features_SVD' : pass1_features_SVD,
             'FeatureMethod': FeatureMethod,
             'Algorithm': ('Ridge', {'n_alphas': (25,)})},
            # {'FeatureScaler': FeatureScaler,
            #  'pass1_features': pass1_features,
            #  'FeatureSelection': FeatureSelection,
            #  'TextTypes': TextTypes,
            #  'pass1_features_SVD' : pass1_features_SVD,
            #  'FeatureMethod': FeatureMethod,
            #  'Algorithm': ('Ridge_multitarget', {'n_alphas': (25,)})},
            # {'FeatureScaler': FeatureScaler,
            #  'pass1_features': pass1_features,
            #  'FeatureSelection': FeatureSelection,
            #  'TextTypes': TextTypes,
            #  'pass1_features_SVD' : pass1_features_SVD,
            #  'FeatureMethod': FeatureMethod,
            #  'Algorithm': ('XGBoost', {'n_estimators': (100,150,300),'max_depth':(2,3)})},
        )

        paramsets = []
        for par in LearnParams:
            parset = Utils.generate_paramset(par,seed=666)
            parset = Utils.simplify_params(parset)
            paramsets += parset

        paramsets = [{'LearnParams':x} for x in paramsets]

        random.Random(1).shuffle(paramsets)

        pickle.dump(paramsets, open(params_file, 'wb'))

    print('\nTotal number of parameters to test: %i. Starting!\n' % len(paramsets))

    current_set=-1
    best_params = {}

    if IS_CLUSTER==0:

        from MainPipeline import MainPipeline

        for current_set in range(0,len(paramsets)):
            par = copy.deepcopy(paramsets[current_set])  # create local copy, will be modified
            print('set %i of %i, %s' % (current_set+1,len(paramsets),Utils.print_parameters(par)))
            my_object = MainPipeline(RootPath=RootPath,ResultPath=ResultPath,RECOMPUTE_ALL=0,RANDOM_TEST=0,SINGLE_THREAD_TEST=0,
                         CV_folds = 8,
                         Type = 'regression',
                         N_workers = 8,
                         verbose_level=1,
                         run_type='looper_local', # local model keeps more data for each set (no need to save memory)
                         dimensions=['infovalue','reliability','subjectivity'],#None,#None,
                         **par
                         )
            if not ALLOW_FAILURE:
                result = my_object.run()
                best_params, was_updated = Utils.get_best_params(best_params, result, k_max=25)
                # save updated results, if any improvement
                if was_updated:
                    print('!!! Found better parameters, updating results !!!\n')
                    pickle.dump((current_set, paramsets, best_params), open(RootPath + 'results' + os.sep + 'looper_best_params' + os.sep + 'all_paramsets.pickle', 'wb'))
                    Utils.save_params(best_params, savepath_root=RootPath + 'results' + os.sep + 'looper_best_params', total_sets=current_set + 1)
            else:
                try:
                    result = my_object.run()
                    best_params,was_updated = Utils.get_best_params(best_params,result,k_max=25)
                    # save updated results, if any improvement
                    if was_updated:
                        print('!!! Found better parameters, updating results !!!\n')
                        pickle.dump((current_set, paramsets, best_params), open(RootPath + 'results' + os.sep + 'looper_best_params' + os.sep + 'all_paramsets.pickle', 'wb'))
                        Utils.save_params(best_params,savepath_root = RootPath + 'results' + os.sep + 'looper_best_params',total_sets = current_set+1)
                except :
                    print('!!!!!! FAILED TO RUN THIS PARAMETER SET: %s !!!!!!\n' % str(sys.exc_info()))

    else:

        k = 0
        kk = 0
        paramsets_batch = []
        currset = []
        for current_set in range(0, len(paramsets)):

            paramsets[current_set]['RootPath'] = RootPath
            paramsets[current_set]['ResultPath'] = ResultPath

            currset.append(paramsets[current_set])
            if len(currset) == SETS_PER_JOB or current_set == len(paramsets) - 1:
                kk += 1
                file = ResultPath + 'parameter_batch%i' % (kk)
                paramsets_batch.append({'params': currset, 'file': file})
                k += len(currset)
                currset = []

        assert k == len(paramsets), 'BUG!'

        skipped = 0
        paramsets_batch_new = []
        for k in range(len(paramsets_batch)):

            result_pickle = paramsets_batch[k]['file'] + '_RESULTS.pickle'
            file_sh = paramsets_batch[k]['file'] + '_job.sh'

            if os.path.isfile(result_pickle) and os.path.isfile(file_sh):
                print('skipping batch %i (files already found)' % (k + 1))
                skipped += 1
                continue

            paramsets_batch[k]['results_filename'] = result_pickle
            paramsets_batch[k]['file_sh'] = file_sh

            with open(paramsets_batch[k]['file_sh'], 'w', encoding='utf-8') as f:
                f.write('#!/bin/bash\n')
                f.write('#SBATCH -p short\n')
                f.write('#SBATCH -t 00:90:00\n')
                f.write('#SBATCH --nodes=1\n')
                f.write('#SBATCH --job-name=looperbatch%i\n' % (k + 1))
                f.write('#SBATCH -o %slooperbatch%i.out\n' % (ResultPath, k + 1))
                f.write('#SBATCH --ntasks=1\n')
                f.write('#SBATCH --cpus-per-task=10\n')
                f.write('#SBATCH --mem-per-cpu=2G\n')

                f.write('module load Python/3.6.1-goolfc-triton-2017a\n')
                #f.write('module load numpy/1.11.1-goolf-triton-2016b-Python-3.5.1\n')
                #f.write('module load scipy/0.18.0-goolf-triton-2016b-Python-3.5.1 \n')

                f.write('export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n')
                f.write('srun -c $SLURM_CPUS_PER_TASK python3 param_looper_run.py %s %i\n' % (params_file_batch,k))

            #pickle.dump((paramsets_batch[k]['params'], paramsets_batch[k]['results_filename']), open(file_pickle, 'wb'))
            paramsets_batch_new.append(paramsets_batch[k])

        assert len(paramsets_batch) == len(paramsets_batch_new) + skipped

        pickle.dump(paramsets_batch, open(params_file_batch, 'wb'))

        paramsets_batch = paramsets_batch_new

        paramsets_batch = paramsets_batch[0:min(MAX_JOBS, len(paramsets_batch))]

        for k in range(0, len(paramsets_batch)):

            print('sending batch %i: %s' % (k + 1, paramsets_batch[k]['file_sh']))
            #
            if IS_CLUSTER == 2:
                os.system('qsub %s' % paramsets_batch[k]['file_sh'])
            elif IS_CLUSTER == 1:
                from param_looper_run import function_run
                function_run(params_file_batch,k)
