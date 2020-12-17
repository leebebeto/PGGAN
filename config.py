import argparse

def arg_parse():
    # configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--z_dim', type=int, default=512, help='dimension to latent vector')
    parser.add_argument('--rindex', type=int, default=0, help='resolution index')
    parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--stab_iter', type=int, default=16000, help='number of iterations for fadein/stabilize')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--ndf', type=int, default=512, help='discriminator dimension')
    args = parser.parse_args()

    print(args)
    return args