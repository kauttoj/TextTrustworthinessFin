# loop over parameter candidates and keep track of best test results (smallest MSE)
import pickle
import sys
from MainPipeline import MainPipeline
import time
import os

def function_run(file,index):
    starttime = time.time()
    print('Loading parameter file \'%s\' with index %i' % (file,index))
    params_dict = pickle.load(open(file, 'rb'))
    params = params_dict[index]['params']
    resultfile = params_dict[index]['results_filename']
    if not os.path.isfile(resultfile):
        results=[]
        for k,param in enumerate(params):
            my_object = MainPipeline(RECOMPUTE_ALL=0, RANDOM_TEST=0, SINGLE_THREAD_TEST=0,
                                     CV_folds=10,
                                     Type='regression',
                                     N_workers=10,
                                     verbose_level=0,
                                     run_type='looper',
                                     dimensions='all',  # None,
                                     **param,
                                     )
            print('... STARTING batch %i of %i' % (k+1,len(params)))
            print('\nparameters:' + str(param) + '\n')
            res = my_object.run()
            results.append(res)
            pickle.dump(results, open(resultfile, 'wb'))

        elapsedtime= (time.time() - starttime)/60.0
        print('all done! Took %0.1f minutes (%0.2f per set)' % (elapsedtime,elapsedtime/len(params)))

    else:
        print('resultfile already present, exiting...')

if __name__ == "__main__":
    file = sys.argv[1:][0]
    index = int(sys.argv[1:][1])
    #print('arg = %s' % file)
    function_run(file,index)


