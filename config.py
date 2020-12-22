import argparse

def arg_parse():
    # configuration
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--z_dim', type=int, default=512, help='dimension to latent vector')
    parser.add_argument('--ngf', type=int, default=512, help='generator dimension')
    parser.add_argument('--ndf', type=int, default=512, help='discriminator dimension')

    # train
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--rindex', type=int, default=0, help='resolution index')
    parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--stab_iter', type=int, default=16000, help='number of iterations for fadein/stabilize')
    parser.add_argument('--retrain', type=bool, default=False, help='whether to retrain')
    parser.add_argument('--retrain_resol', type=int, default=0, help='retraining resolution')

    # logging
    parser.add_argument('--print_freq', type=int, default=500, help='frequency for print')
    parser.add_argument('--save_image_freq', type=int, default=1000, help='frequency for saving images')
    parser.add_argument('--save_model_freq', type=int, default=1000, help='frequency for saving model checkpoint')
    args = parser.parse_args()

    print(args)
    return args