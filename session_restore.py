from model.srlIdConfig import SrlIdConfig
from model.srl_model import SRLModel
from model.base_model import BaseModel
from model.predIdConfig import PredIdConfig

config = SrlIdConfig()
# baseModel = BaseModel(config)
# baseModel.close_session()
model = SRLModel(config)
model.build()
# model.close_session()
config.dir_model = config.dir_model_root + "slstm_lr0.05_batch2_layer2" + "/"
model.restore_session(config.dir_model)


