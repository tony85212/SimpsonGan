import argparse
from dataset import *
from model import *
from simpsongan import *


if __name__ == '__main__':
    #parser
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help = "data path")
    parser.add_argument("mode", choices=['train', 'test', 'custom'], help ="train, test, customize")
    parser.add_argument("--epoch", type = int, default = 50, help="epoch")
    parser.add_argument("--size", type = int, default = 2000, help="datasize")
    parser.add_argument("--model", choices=['ga', 'gb', 'da', 'db'], default = 'ga', help="model")
    args = parser.parse_args()
    #making directory
    make_directory("model/")
    make_directory("result/")
    make_directory("tmp/")
    #difine parameters
    params = {
        'num_epochs':args.epoch,
        'decay_epoch':100,
        'ngf':32,   #number of generator filters
        'ndf':64,   #number of discriminator filters
        'num_resnet':6, #number of resnet blocks
        'lrG':0.0002,    #learning rate for generator
        'lrD':0.0002,    #learning rate for discriminator
        'beta1':0.5 ,    #beta1 for Adam optimizer
        'beta2':0.999 ,  #beta2 for Adam optimizer
        'lambdaA':10 ,   #lambdaA for cycle loss
        'lambdaB':10  ,  #lambdaB for cycle loss
        'lambdaC':10 ,   #lambdaC for eyes cycle loss
        'lambdaD':10  ,  #lambdaD for mouth cycle loss
        'eyes_weight':0.5 ,  #weight loss of eyes
        'mouth_weight':0.5 , #weight loss of mouth
    }
    print(params['lambdaA'], params['lambdaB'], params['lambdaC'], params['lambdaD'], params['eyes_weight'], params['mouth_weight'])
    #initialize simpsongan
    print(" [*] Initializing SimpsonGan")
    gan = SimpsonGan(params)
    #train & test
    if args.mode == 'train':
        print(" [*] Training start!")
        trainA = load_image(args.data + '/trainA', args.size)
        trainB = load_image(args.data + '/trainB', args.size)
        gan.train(trainA, trainB)
        print(" [*] Training finished!")
    elif args.mode == 'test':
        print(" [*] Test start!")
        testA = load_image(args.data + '/testA', args.size)
        testB = load_image(args.data + '/testB', args.size)
        gan.test(testA, testB)
        print(" [*] Test finished!")
    else:
        print(" [*] Custom mode!")
        gan.custom(args.data, args.model)
