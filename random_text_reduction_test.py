# run code with best parameters (one set for each dimension)

import pickle
import os
from MainPipeline import MainPipeline
import Utils
import copy

if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

    params={}

    params['reliability'] = (
        {'TextTypes': ['LEMMA'], 'Algorithm': ('Ridge', {'n_alphas': 25}), 'pass1_features': 200, 'FeatureSelection': ['fisher', 'global', 800], 'FeatureScaler': 'MaxAbsScaler', 'FeatureMethod': ['BOW_3', 'TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'], 'pass1_features_SVD': None},
        )

    RootPath = r'D:/JanneK/Documents/git_repos/LaureaTextAnalyzer/'

    CV_folds=7

    all_results=[]

    for culling_rate in (0.6,0.5,0.40,0.30,0.20,0.10,0):
        culrate=[]

        print('!!!!!! Culling rate = %f\n' % culling_rate)
        for iter in range(10):
            for dim in set(params.keys()):#.intersection(('reliability',)):

                par = copy.deepcopy(params[dim])

                print('Dimension = %s, %s' % (dim, Utils.print_parameters(par)))

                my_object = MainPipeline(RootPath=RootPath,RECOMPUTE_ALL=0,RANDOM_TEST=0,SINGLE_THREAD_TEST=0,
                             CV_folds = CV_folds,
                             Type = 'regression',
                             dimensions=dim,
                             verbose_level=1,
                             N_workers=7,
                             run_type='simple',
                             culling_rate=culling_rate,
                             **par,
                             )
                result = my_object.run()
                culrate.append(result)
            if culling_rate==0:
                break

        all_results.append(culrate)
