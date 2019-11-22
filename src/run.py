import gan as gan
import argparse

def parse_args():
    desc = 'GAN cost function comparison'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--epochs',type=int,default=1,help='Number of epochs to train model')
    parser.add_argument('--gpus',type=str,default='3',help='Commaseparated indices of GPUs to use (0-ind)')
    parser.add_argument('--dataset',type=str,default='cifar',help='Dataset to use (mnist,cifar)')
    parser.add_argument('--d_net',type=str,default='snconv',help='Discriminator network (dense,conv,sndense,snconv)')
    parser.add_argument('--g_net',type=str,default='snconv',help='Generator network (dense,conv,sndense,snconv)')
    parser.add_argument('--m_dim',type=int,default=64,help='Model complexity factor')
    parser.add_argument('--z_dim',type=int,default=128,help='Latent space (z) dimension')

    parser.add_argument('--batch_size',type=int,default=64,help='Batch size')
    parser.add_argument('--opt',type=str,default='adam',help='Optimizer for gradient descent (adam, sgd)')
    
    parser.add_argument('--g_lr',type=float,default=1e-4,help='Generator learning rate')
    parser.add_argument('--g_beta1',type=float,default=0.5,help='Generator beta1 parameter for Adam optimizer')
    parser.add_argument('--g_beta2',type=float,default=0.999,help='Generator beta2 parameter for Adam optimizer')
    parser.add_argument('--g_adameps',type=float,default=1e-8,help='Generator eps-hat parameter for Adam optimizer')

    parser.add_argument('--g_adamreset',type=int,default=0,help='Generator frequency of resetting Adam optimizer (0 disables)')
    
    parser.add_argument('--d_lr',type=float,default=1e-4,help='Discriminator learning rate')
    parser.add_argument('--d_beta1',type=float,default=0.5,help='Discriminator beta1 parameter for Adam optimizer')
    parser.add_argument('--d_beta2',type=float,default=0.999,help='Discriminator beta2 parameter for Adam optimizer')
    parser.add_argument('--d_adameps',type=float,default=1e-8,help='Discriminator eps-hat parameter for Adam optimizer')
    
    parser.add_argument('--d_cost',type=str,default='ns',help='Cost function for training')
    parser.add_argument('--g_cost',type=str,default='ns',help='Cost function for training')
    parser.add_argument('--g_cost_parameter',type=float,default=0,help='Cost function parameter in (0,1)')
    parser.add_argument('--g_renorm',type=str,default='none',help='Renorm gradient magnitude (none, const, frac, nsat, unit)')
    parser.add_argument('--g_layers',type=int,default=4,help='Total layers in dense generator')
    parser.add_argument('--d_layers',type=int,default=4,help='Total layers in dense discriminator')
    parser.add_argument('--g_sn',type=int,default=0,help='Use spectral normalization for G (if net supports it)')
    parser.add_argument('--d_sn',type=int,default=0,help='Use spectral normalization for D (if net supports it)')
    parser.add_argument('--g_sa',type=int,default=0,help='Use self attention for G (only supported by dcg net)')
    parser.add_argument('--d_sa',type=int,default=0,help='Use self attention for D (only supported by dcg net)')
    
    parser.add_argument('--metrics',type=int,default=1,help='Run metrics (disable for speed or lacking networks)')
    parser.add_argument('--fid_n',type=int,default=50000,help='Samples used to calculate FID')
    parser.add_argument('--eval_n',type=int,default=20,help='Number of evaluation steps (time intensive)')
    parser.add_argument('--eval_skip',type=int,default=0,help='Skip FID and CDD during all except final eval (time saver)')
    parser.add_argument('--runs_n',type=int,default=20,help='Number of times to train GAN')
    parser.add_argument('--output_folder',type=str,default='debug',help='Folder to save outputs')

    return parser.parse_args()

def main():
    a = parse_args()
    of = a.output_folder
    for i in xrange(a.runs_n):
        vars(a)['output_folder'] = 'out/' + of + '_' + str(i)
        gan.main(a)

if __name__ == '__main__': main()
