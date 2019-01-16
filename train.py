import sys

import utils

from parameters import *
from trainer import Trainer


if __name__ == '__main__':
    config = get_parameters()
    config.command = 'python ' + ' '.join(sys.argv)
    print(config)
    trainer = Trainer(config)
    trainer.train()
    utils.save_ckpt(trainer, final=True)
