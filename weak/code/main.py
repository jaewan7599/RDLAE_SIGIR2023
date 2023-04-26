import world
import utils
import Procedure

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================

import register
from register import dataset


print(f"Current model: {world.model_name}")
Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)

Procedure.Test(dataset, Recmodel, 0)