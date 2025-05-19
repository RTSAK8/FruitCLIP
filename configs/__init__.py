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


def get_mocoop_cfg(cfg: CN):
    """
    Get the config for the MoCoOp model.
    """
    cfg.TRAINER.MoCoOp = CN()
    cfg.TRAINER.MoCoOp.ENABLE = True  # LARS loss

    cfg.TRAINER.MoCoOp.TEXT_LOSS_WEIGHT = 1.0  # weighf of LARS text-to-text loss
    cfg.TRAINER.MoCoOp.GATE_LOSS_WEIGHT = 1.0  # weighf of LARS text-to-text loss
    cfg.TRAINER.MoCoOp.N_CTX = 16  # number of context vectors
    cfg.TRAINER.MoCoOp.CTX_INIT = []  # initialization words
    cfg.TRAINER.MoCoOp.PREC = "amp"  # fp16, fp32, amp

    cfg.TRAINER.MoCoOp.ENABLE_CORRECTION = False
    cfg.TRAINER.MoCoOp.ENABLE_IMPLICIT_OP = "sum"  # mul
    cfg.TRAINER.MoCoOp.PRETRAINED_PROMPTS_DIR = None
    cfg.TRAINER.MoCoOp.TRAIN_W = True
    cfg.TRAINER.MoCoOp.FINETUNE_VIT_LN = True
