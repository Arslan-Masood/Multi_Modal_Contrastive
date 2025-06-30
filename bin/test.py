import sys 
sys.path.insert(1, '/scratch/work/masooda1/Multi_Modal_Contrastive/mocop')
import os
from typing import Dict, Union

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

import dataset
import model
from testing import test


@hydra.main(config_path="/scratch/work/masooda1/Multi_Modal_Contrastive/configs", config_name="train.yml")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    best = test(cfg)


if __name__ == "__main__":
    main()
