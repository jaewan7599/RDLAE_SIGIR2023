'''
Design training and test process
'''
import world
import numpy as np
import torch
import utils
import multiprocessing
from time import perf_counter

CORES = multiprocessing.cpu_count() // 2


def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    
    r = utils.getLabel(groundTrue, sorted_items)
    
    pre, recall, ndcg = [], [], []

    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
        
    return {'precision':np.array(pre),
            'recall':np.array(recall),
            'ndcg':np.array(ndcg),
            }

            
def Test(dataset, Recmodel, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict

    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)

    if multicore == 1:
        pool = multiprocessing.Pool(CORES)

    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks)),
               }

    with torch.no_grad():
        users = list(testDict.keys())
        rating_list = []
        groundTrue_list = []
        t = perf_counter()

        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            exclude_index = []
            exclude_items = []

            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)

            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)

            rating = rating.cpu().numpy()
            del rating
            
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)

        inference_time = perf_counter()-t

        print("Inference time: {:.4f}s".format(inference_time))
                    
        X = zip(rating_list, groundTrue_list)

        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))

        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
            
        results['precision'] /= float(len(users))
        results['recall'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        
        if multicore == 1:
            pool.close()

        print(f"Training time: {round(Recmodel.train_time)}")
        print(results)
        
    return results
