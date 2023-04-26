import world
import utils

import Procedure
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset
import numpy as np


print(f"Current model: {world.model_name}")
item_freq = np.array(dataset.UserItemNet.sum(axis=0)).squeeze()
world.pscore = np.maximum((item_freq / item_freq.max()) ** 0.5, 10e-3)

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)

# init tensorboard
Procedure.Test(dataset, Recmodel)
