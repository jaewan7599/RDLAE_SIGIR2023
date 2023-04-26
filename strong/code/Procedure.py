'''
Design training and test process
'''
import world
import numpy as np
import torch
import utils
import multiprocessing
from time import perf_counter
import math

CORES = multiprocessing.cpu_count() // 2


def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    head_item = X[2]
    tail_item = X[3]
    head_groundTrue = []
    tail_groundTrue = []
    
    for gt in groundTrue:
        head_groundTrue.append(list(filter(lambda x: x in head_item, gt)))
        tail_groundTrue.append(list(filter(lambda x: x in tail_item, gt)))

    r = utils.getLabel(groundTrue, sorted_items)
    head_r = utils.getLabel(head_groundTrue, sorted_items)
    tail_r = utils.getLabel(tail_groundTrue, sorted_items)
    
    pre, recall, ndcg = [], [], []
    pre_head, recall_head, ndcg_head = [], [], []
    pre_tail, recall_tail, ndcg_tail = [], [], []
    upre, urecall, undcg = [], [], []
    
    pscore = world.pscore

    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
        
        ret_head = utils.RecallPrecision_ATk(head_groundTrue, head_r, k)
        pre_head.append(ret_head['precision'])
        recall_head.append(ret_head['recall'])
        ndcg_head.append(utils.NDCGatK_r(head_groundTrue, head_r, k))
        
        ret_tail = utils.RecallPrecision_ATk(tail_groundTrue, tail_r, k)
        pre_tail.append(ret_tail['precision'])
        recall_tail.append(ret_tail['recall'])
        ndcg_tail.append(utils.NDCGatK_r(tail_groundTrue, tail_r, k))

        ret = utils.uRecPrecatK_r(sorted_items, groundTrue, r, k, pscore)
        upre.append(ret['uprecision'])
        urecall.append(ret['urecall'])
        undcg.append(utils.uNDCGatK_r(sorted_items, groundTrue, r, k, pscore))

    return {'precision':np.array(pre),
            'recall':np.array(recall),
            'ndcg':np.array(ndcg),
            
            'precision(head)':np.array(pre_head),
            'recall(head)':np.array(recall_head),
            'ndcg(head)':np.array(ndcg_head),
            
            'precision(tail)':np.array(pre_tail),
            'recall(tail)':np.array(recall_tail),
            'ndcg(tail)':np.array(ndcg_tail),
            
            'urecall':np.array(urecall), 
            'uprecision':np.array(upre), 
            'undcg':np.array(undcg),
            }


def get_valid_score(Recmodel, dataset):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    validDict: dict = dataset.validDict

    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = 100

    with torch.no_grad():
        users = list(validDict.keys())
        users_list = []
        rating_list = []
        groundTrue_list = []

        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getValidUserPosItems(batch_users)
            groundTrue = [validDict[u] for u in batch_users]

            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getvalidUsersRating(batch_users_gpu)
            exclude_index = []
            exclude_items = []

            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)

            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)

            rating = rating.cpu().numpy()
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        
        X = zip(rating_list, groundTrue_list)

        ndcg, undcg = 0, 0

        for x in X:
            sorted_items = x[0].numpy()
            groundTrue = x[1]
            r = utils.getLabel(groundTrue, sorted_items)
            ndcg += utils.NDCGatK_r(groundTrue, r, 100)
            undcg += utils.uNDCGatK_r(sorted_items, groundTrue, r, 100, world.pscore)

        ndcg /= dataset.num_valid_user
        undcg /= dataset.num_valid_user

    Recmodel = Recmodel.train()

    return ndcg, undcg


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
               
               'precision(head)': np.zeros(len(world.topks)),
               'recall(head)': np.zeros(len(world.topks)),
               'ndcg(head)': np.zeros(len(world.topks)),
               
               'precision(tail)': np.zeros(len(world.topks)),
               'recall(tail)': np.zeros(len(world.topks)),
               'ndcg(tail)': np.zeros(len(world.topks)),
               
               'uprecision': np.zeros(len(world.topks)),
               'urecall': np.zeros(len(world.topks)),
               'undcg': np.zeros(len(world.topks)),
               }

    with torch.no_grad():
        users = list(testDict.keys())
        users_list = []
        rating_list = []
        groundTrue_list = []
        t = perf_counter()
        
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getTestUserPosItems(batch_users)
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
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)

        inference_time = perf_counter()-t

        print("Inference time: {:.4f}s".format(inference_time))
        num_items = dataset.m_items
        
        item_counts = np.array(dataset.UserItemNet.sum(axis=0)).squeeze()
        sorted_by_item_counts = np.argsort(item_counts)[::-1]
        divide_idx = math.ceil(num_items * 0.2)
        
        head_item = sorted_by_item_counts[:divide_idx]
        tail_item = sorted_by_item_counts[divide_idx:]
        rep_head_item = np.repeat([head_item], repeats=len(rating_list), axis=0)
        rep_tail_item = np.repeat([tail_item], repeats=len(rating_list), axis=0)

        head_num, tail_num = 0, 0
        for gt_list in groundTrue_list:
            for gt in gt_list:
                if len(list(filter(lambda x: x in head_item, gt))) != 0:
                    head_num += 1
                if len(list(filter(lambda x: x in tail_item, gt))) != 0:
                    tail_num += 1
        
        X = zip(rating_list, groundTrue_list, rep_head_item, rep_tail_item)

        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))

        for result in pre_results:
            results['precision'] += result['precision']
            results['recall'] += result['recall']
            results['ndcg'] += result['ndcg']
            
            results['precision(head)'] += result['precision(head)']
            results['recall(head)'] += result['recall(head)']
            results['ndcg(head)'] += result['ndcg(head)']
            
            results['precision(tail)'] += result['precision(tail)']
            results['recall(tail)'] += result['recall(tail)']
            results['ndcg(tail)'] += result['ndcg(tail)']
            
            results['urecall'] += result['urecall']
            results['uprecision'] += result['uprecision']
            results['undcg'] += result['undcg']

        results['precision'] /= dataset.num_test_user
        results['recall'] /= dataset.num_test_user
        results['ndcg'] /= dataset.num_test_user
        
        results['precision(head)'] /= head_num
        results['recall(head)'] /= head_num
        results['ndcg(head)'] /= head_num
        
        results['precision(tail)'] /= tail_num
        results['recall(tail)'] /= tail_num
        results['ndcg(tail)'] /= tail_num
        
        results['urecall'] /= dataset.num_test_user
        results['uprecision'] /= dataset.num_test_user
        results['undcg'] /= dataset.num_test_user

        if multicore == 1:
            pool.close()

        print(f"Training time: {round(Recmodel.train_time)}, Valid NDCG@100: {np.round(Recmodel.valid_ndcg, 4)}")
        print(results)

    return results
