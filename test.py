import sys

import utils

from parameters import *
from sagan_models import Generator, Discriminator


if __name__ == '__main__':
    config = get_parameters()
    config.command = 'python ' + ' '.join(sys.argv)
    print(config)
    utils.check_for_CUDA(config)

    # Load pretrained model (if provided)
    if config.pretrained_model != '':
        utils.load_pretrained_model(config)
    else:
        assert config.num_of_classes, "Please provide number of classes! Eg. python3 test.py --num_of_classes 10"
        config.G = Generator(config.z_dim, config.g_conv_dim, config.num_of_classes).to(config.device)
        config.D = Discriminator(config.d_conv_dim, config.num_of_classes).to(config.device)

    config.G.eval()
    config.D.eval()
    print(config.G, config.D)
