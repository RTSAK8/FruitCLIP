from yacs.config import CfgNode as CN


def get_coop_cfg(cfg: CN):
    """
    Get the config for the CoOp model.
    """
    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'


def get_cocoop_cfg(cfg: CN):
    """
    Get the config for the CoCoOp model.
    """
    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp


def get_lasp_cfg(cfg: CN):
    """
    Get the config for the LASP model.
    """
    cfg.TRAINER.LASP = CN()
    cfg.TRAINER.LASP.ENABLE = True # LARS loss
    cfg.TRAINER.LASP.LASP_PROMPTS = ['a photo of {}'] # List of textual prompts for LARS
    cfg.TRAINER.LASP.LASP_LOSS_WEIGHT = 1.0 # weighf of LARS text-to-text loss
    cfg.TRAINER.LASP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.LASP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.LASP.PREC = "amp"  # fp16, fp32, amp

    cfg.TRAINER.LASP.ENABLE_CORRECTION = False
    cfg.TRAINER.LASP.ENABLE_IMPLICIT_OP = 'sum' # mul
    cfg.TRAINER.LASP.PRETRAINED_PROMPTS_DIR = ''
    cfg.TRAINER.LASP.TRAIN_W = True
    cfg.TRAINER.LASP.FINETUNE_VIT_LN = True

    cfg.DATASET.INCLUDE_ALL_CLASSES = True