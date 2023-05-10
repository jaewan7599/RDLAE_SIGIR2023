import numpy as np
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import world
from world import cprint


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems


def load_strong_data(input_data, target_data, prev_max_item):
    num_user = 0
    targetUniqueUsers, targetItem, targetUser = [], [], []
    inputUniqueUsers, inputItem, inputUser = [], [], []
    m_item = prev_max_item
    DataSize = 0

    with open(target_data) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                try:
                    items = [int(i) for i in l[1:]]
                except Exception:
                    continue
                uid = int(l[0])
                targetUniqueUsers.append(uid)
                targetUser.extend([uid] * len(items))
                targetItem.extend(items)
                m_item = max(m_item, max(items))
                num_user = max(num_user, uid)
                DataSize += len(items)

    targetUniqueUsers = np.array(targetUniqueUsers)
    targetUser = np.array(targetUser)
    targetItem = np.array(targetItem)

    with open(input_data) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                try:
                    items = [int(i) for i in l[1:]]
                except Exception:
                    continue
                uid = int(l[0])
                inputUniqueUsers.append(uid)
                inputUser.extend([uid] * len(items))
                inputItem.extend(items)
                m_item = max(m_item, max(items))

    num_user += 1
    inputUniqueUsers = np.array(inputUniqueUsers)
    inputUser = np.array(inputUser)
    inputItem = np.array(inputItem)

    inputUserItemNet = csr_matrix((np.ones(len(inputUser)), (inputUser, inputItem)),
                                    shape=(num_user, m_item+1))
    
    return num_user, m_item, DataSize, inputUserItemNet, targetUser, targetItem


class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """
    def __init__(self, path):
        # train or test
        cprint(f'loading [{path}]')
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.txt'

        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []

        self.traindataSize = 0
        self.validDataSize = 0
        self.testDataSize = 0
        
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    try:
                        items = [int(i) for i in l[1:]]
                    except Exception:
                        continue
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)

        self.n_user += 1
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        self.num_valid_user, self.m_item, self.validDataSize, self.validUserItemNet, self.validUser, self.validItem = \
            load_strong_data(path + '/valid_in.txt', path + '/valid.txt', self.m_item)

        self.num_test_user, self.m_item, self.testDataSize, self.testUserItemNet, self.testUser, self.testItem = \
            load_strong_data(path + '/test_in.txt', path + '/test.txt', self.m_item)
            
        self.m_item += 1
        
        self.Graph = None
        print(f'{self.n_user} training users, {self.num_valid_user} valid users, {self.num_test_user} test users, {self.m_item} items')
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.validDataSize} interactions for validation")
        print(f"{self.testDataSize} interactions for testing")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))

        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.

        # pre-calculate
        self.__validDict = self.__build_valid()
        self.__testDict = self.__build_test()

        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def validDict(self):
        return self.__validDict

    @property
    def testDict(self):
        return self.__testDict
    
    @property
    def allPos(self):
        return self._allPos

    def __build_valid(self):
        """
        return:
            dict: {user: [items]}
        """
        valid_data = {}
        for i, item in enumerate(self.validItem):
            user = self.validUser[i]
            if valid_data.get(user):
                valid_data[user].append(item)
            else:
                valid_data[user] = [item]
        return valid_data

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserPosItems(self, users):
        posItems = []
        
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    
    def getValidUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.validUserItemNet[user].nonzero()[1])
        return posItems

    def getTestUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.testUserItemNet[user].nonzero()[1])
        return posItems
