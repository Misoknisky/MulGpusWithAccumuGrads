#coding=utf-8
'''
Created on 2018年3月4日
@author: Administrator
@email: 1113471782@qq.com
'''

import sys
sys.path.append("../")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import argparse
import logging
from main import train,evaluate,predict,prepare

def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Reading Comprehension on BaiduRC dataset')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--gpus', nargs='+', default=['0','1'],help='specify gpus device')
    parser.add_argument('--accumulate_n',type=int,default=1,help="accumulate gradents")
    extra_settings = parser.add_argument_group('extra settings')
    # loss选择
    extra_settings.add_argument('--use_multi_ans_loss', type=bool, default=True,
                                help='whether answer prediction with multi-answer')
    extra_settings.add_argument('--multi_ans_norm', choices=['None', 'max_min', 'sum'], default='None',
                                help='how to normalize the multi ans')
    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=1,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_size', type=int, default=2,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=10,
                                help='train epochs')
    train_settings.add_argument('--evaluate_cnt_in_one_epoch', type=int, default=2,
                                help='evaluate count in one epoch, default 0, evaluate for epoch (must >0)')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', choices=['MBIDAF','BIDAF', 'MLSTM',"DCA"], default='BIDAF',
                                help='choose the algorithm to use')
    model_settings.add_argument('--embed_size', type=int, default=300,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=150,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--max_p_num', type=int, default=5,
                                help='max passage num in one sample')
    model_settings.add_argument('--max_p_len', type=int, default=500,
                                help='max length of passage')
    model_settings.add_argument('--max_q_len', type=int, default=60,
                                help='max length of question')
    model_settings.add_argument('--max_a_len', type=int, default=300,
                                help='max length of answer')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--pretrained_word_path',default="data/vectors.txt")
    path_settings.add_argument('--train_files', nargs='+',
                               default=['data/train.json'],
                               help='list of files that contain the preprocessed train data')
    path_settings.add_argument('--dev_files', nargs='+',
                               default=['data/dev.json'],
                               help='list of files that contain the preprocessed dev data')
    path_settings.add_argument('--test_files', nargs='+',
                               default=['data/test.json'],
                               help='list of files that contain the preprocessed test data')
    path_settings.add_argument('--brc_dir', default='data/baidu',
                               help='the dir with preprocessed baidu reading comprehension data')
    path_settings.add_argument('--vocab_dir', default='data/vocab/',
                               help='the dir to save vocabulary')
    path_settings.add_argument('--model_dir', default='data/models/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='data/results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='data/summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')
    return parser.parse_args()




def run():
    """
    Prepares and runs the whole system.
    """
    args = parse_args()

    logger = logging.getLogger("brc")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
   # string=",".join([g_num for g_num in args.gpus])
   # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
   # os.environ["CUDA_VISIBLE_DEVICES"] = string
    logger.info('Running with args : {}'.format(args))
    if args.prepare:
        prepare(args)
    if args.train:
        train(args)
    if args.evaluate:
        evaluate(args)
    if args.predict:
        predict(args)

if __name__ == '__main__':
    run()
