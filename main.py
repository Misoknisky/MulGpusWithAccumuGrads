#coding=utf-8
'''
Created on 2020年3月4日
@author: Administrator
@email: 1113471782@qq.com
'''
import os
import logging
import pickle
from tqdm import tqdm
import json
from utils.dataset import BRCDataset
from utils.vocab import Vocab
from utils.optimizer import create_train_op, average_gradients
from utils.accumulate_steps import AccumulateSteps
import tensorflow as tf
from model import Model
from utils.dureader_eval import normalize
from utils.dureader_eval import compute_bleu_rouge

def print_variable_summary():
    import pprint
    variables = sorted([[v.name, v.get_shape()] for v in tf.global_variables()])
    pprint.pprint(variables)
def prepare(args):
    logger = logging.getLogger("brc")
    logger.info('Checking the data files...')
    for data_path in args.train_files + args.dev_files + args.test_files:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    logger.info('Preparing the directories...')
    for dir_path in [args.vocab_dir, args.model_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    logger.info('Building vocabulary...')
    brc_data = BRCDataset(args.max_p_num, 
                          args.max_p_len, 
                          args.max_q_len,
                          args.gpus,
                          args.batch_size,
                          args.train_files, 
                          args.dev_files, 
                          args.test_files)
    vocab = Vocab(init_random=False,trainable_oov_cnt_threshold=2)
    for word in brc_data.word_iter('train'):
        vocab.add(word)
    unfiltered_vocab_size = vocab.size()
    vocab.filter_tokens_by_cnt(min_cnt=2)
    filtered_num = unfiltered_vocab_size - vocab.size()
    logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num,
                                                                            vocab.size()))
    logger.info('Assigning embeddings...')
#     vocab.build_embedding_matrix(args.pretrained_word_path)
    vocab.randomly_init_embeddings(args.embed_size)
    logger.info('Saving vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'wb') as fout:
        pickle.dump(vocab, fout)
    logger.info('Done with preparing!')


def train(args):
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    brc_data = BRCDataset(args.max_p_num, 
                          args.max_p_len, 
                          args.max_q_len,
                          args.gpus,
                          args.batch_size,
                          args.train_files, 
                          args.dev_files)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    opt=create_train_op(args.optim,args.learning_rate)
    tower_grads,models= [],[]
    train_perplexity = tf.get_variable(
            'train_perplexity', [],
            initializer=tf.constant_initializer(0.0), trainable=False)
    global_step = tf.get_variable(
            'global_step', [],
            dtype=tf.int32,
            initializer=tf.constant_initializer(0), trainable=False)
    for k,gpu_num in enumerate(args.gpus):
        resuse_flag=True if k > 0 else False
        with tf.device('/gpu:%s' % gpu_num):
            with tf.variable_scope('model', reuse=resuse_flag):
                model=Model(vocab,args)
                models.append(model)
                loss=model.loss
                grads=opt.compute_gradients(loss,aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
                for g,v in grads:
                    print(g,v)
                tower_grads.append(grads)
                train_perplexity += loss
   # print_variable_summary()
    ave_grads=average_gradients(tower_grads)
    train_perplexity = train_perplexity / len(args.gpus)
    accumulator=AccumulateSteps(grads_vars=ave_grads,accumulate_step=args.accumulate_n)
    train_op = opt.apply_gradients(accumulator.grads_holder, global_step=global_step)
    #init session
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    pad_id=vocab.get_id(vocab.pad_token)
    max_rouge_l=-1
    for epoch in range(1,args.epochs + 1):
        epoch_sample_num, epoch_total_loss = 0, 0
        real_batch=args.batch_size * len(args.gpus)
        train_batches = brc_data.gen_mini_batches('train',real_batch, pad_id, shuffle=True)
        for batch_num,batch_data in enumerate(train_batches,1):
            passage_num=len(batch_data['passage_token_ids']) //real_batch
            feed_dict=dict()
            for k in range(len(args.gpus)):
                start =k * args.batch_size
                end = (k+1) * args.batch_size
                feed_dict.update({models[k].p: batch_data['passage_token_ids'][start*passage_num:end*passage_num],
                             models[k].q: batch_data['question_token_ids'][start*passage_num:end*passage_num],
                             models[k].p_length: batch_data['passage_length'][start*passage_num:end*passage_num],
                             models[k].q_length: batch_data['question_length'][start*passage_num:end*passage_num],
                             models[k].start_label:batch_data['start_ids'][start:end],
                             models[k].end_label:batch_data['end_ids'][start:end],
                             models[k].match_score:batch_data['match_scores'][start:end]})
            grads_values,loss=sess.run([ave_grads,train_perplexity],feed_dict)
            grads_feed=accumulator.add_grads(grads_values)
            epoch_total_loss += loss * len(batch_data['raw_data'])
            epoch_sample_num += len(batch_data['raw_data'])
            if grads_feed is not None:
                _=sess.run([train_op],feed_dict=grads_feed)
                global_step = tf.train.get_global_step().eval(session=sess)
            if global_step % 1000==0:
                logger.info("Evaluating the model")
                if brc_data.dev_set is not None:
                    eval_batches = brc_data.gen_mini_batches('dev', args.batch_size, pad_id, shuffle=False)
                    eval_loss, bleu_rouge=evaluate(args, eval_batches, vocab,sess)
                    logger.info('Dev eval loss {}'.format(eval_loss))
                    logger.info('Dev eval result: {}'.format(bleu_rouge))
                    if bleu_rouge['Rouge-L'] > max_rouge_l:
                        save(saver,sess,args.model_dir,args.algo)
                        max_rouge_l = bleu_rouge['Rouge-L']
                    logger.info("the epoch {} batch {} rouge-l value is {}".format(epoch,batch_num,bleu_rouge['Rouge-L']))
                    logger.info("the max rouge-l value is {}".format(max_rouge_l))
        if epoch_sample_num % (args.batch_size * len(args.gpus)*args.accumulate_n) !=0:
            grads_feed=accumulator.add_grads(grad_vars=None,right_grads=True)
            _=sess.run([train_op],feed_dict=grads_feed)
            global_step = tf.train.get_global_step().eval(session=sess)
        logger.info('Average train loss for epoch {} is {}'.format(epoch,1.0*epoch_total_loss/epoch_sample_num))
        if brc_data.dev_set is not None:
            eval_batches = brc_data.gen_mini_batches('dev', args.batch_size, pad_id, shuffle=False)
            eval_loss, bleu_rouge=evaluate(args, eval_batches, vocab,sess)
            logger.info('Dev eval loss {}'.format(eval_loss))
            logger.info('Dev eval result: {}'.format(bleu_rouge))
            if bleu_rouge['Rouge-L'] > max_rouge_l:
                save(saver,sess,args.model_dir,args.algo)
                max_rouge_l = bleu_rouge['Rouge-L']
            logger.info("the epoch {} rouge-l value is {}".format(epoch,bleu_rouge['Rouge-L']))
            logger.info("the max rouge-l value is {}".format(max_rouge_l))
    logger.info('Done with model training!')

def evaluate(args,eval_batches,vocab,sess,result_dir=None,result_prefix=None):
    logger = logging.getLogger("brc")
    total_loss, total_num = 0, 0
    result_prob=[]
    pred_answers, ref_answers = [], []
    with tf.device('/gpu:%s' % args.gpus[0]):
        with tf.variable_scope('model',reuse=True):
            model=Model(vocab,args)
    pp_scores = (0.43, 0.23, 0.16, 0.10, 0.09)
    for _, batch_data in enumerate(eval_batches,1):
        feed_dict = {model.p: batch_data['passage_token_ids'],
                    model.q: batch_data['question_token_ids'],
                    model.p_length: batch_data['passage_length'],
                    model.q_length: batch_data['question_length'],
                    model.start_label:batch_data['start_ids'],
                    model.end_label:batch_data['end_ids'],
                    model.match_score:batch_data['match_scores']}
        start_probs, end_probs, loss = sess.run([model.start_probs,
                                                 model.end_probs, model.loss], feed_dict)
        total_loss += loss * len(batch_data['raw_data'])
        total_num += len(batch_data['raw_data'])
        padded_p_len = len(batch_data['passage_token_ids'][0])
        for sample, start_prob, end_prob in zip(batch_data['raw_data'], start_probs, end_probs):
            start_prob_list=[str(element) for element in list(start_prob)]
            end_prob_list=[str(element) for element in list(end_prob)]
            result_prob.append({"question_id":sample['question_id'],"start_prob":start_prob_list,"end_prob":end_prob_list,'padd_len':padded_p_len})
            best_answer, segmented_pred = find_best_answer(sample, start_prob, end_prob,padded_p_len,args,para_prior_scores=pp_scores)
            pred_answers.append({'question_id': sample['question_id'],
                                         'question_type': sample['question_type'],
                                         'answers': [best_answer],
                                         'entity_answers': [[]],
                                         'yesno_answers': [],
                                         'segmented_question': sample['segmented_question'],
                                         'segmented_answers': segmented_pred})
            if 'segmented_answers' in sample:
                ref_answers.append({'question_id': sample['question_id'],
                                         'question_type': sample['question_type'],
                                         'answers': [''.join(seg_ans) for seg_ans in sample['segmented_answers']],
                                         'entity_answers': [[]],
                                         'yesno_answers': []})
    if result_dir is not None and result_prefix is not None:
        result_file = os.path.join(result_dir, result_prefix + '.json')
        with open(result_file, 'w') as fout:
            for pred_answer in tqdm(pred_answers):
                fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')
        prob_file = os.path.join(result_dir,result_prefix+ 'probs.json')
        with open(prob_file,'w') as f:
            for prob in tqdm(result_prob):
                f.write(json.dumps(prob,ensure_ascii=False) + "\n")
        logger.info('Saving {} results to {}'.format(result_prefix, result_file))
    ave_loss = 1.0 * total_loss / total_num
    if len(ref_answers) > 0:
        pred_dict, ref_dict = {}, {}
        for pred, ref in zip(pred_answers, ref_answers):
            question_id = ref['question_id']
            if len(ref['answers']) > 0:
                pred_dict[question_id] = normalize(pred['answers'])
                ref_dict[question_id] = normalize(ref['answers'])
        bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
    else:
        bleu_rouge = None
    return ave_loss, bleu_rouge            
def predict(args):
    """
    predicts answers for test files
    """
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        logger.info('load vocab from {}'.format(os.path.join(args.vocab_dir, 'vocab.data')))
        vocab = pickle.load(fin)
    assert len(args.test_files) > 0, 'No test files are provided.'
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          test_files=args.test_files)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    logger.info('Restoring the model...')
    with tf.device('/gpu:%d' % args.gpus[0]):
        with tf.variable_scope('model'):
            model=Model(vocab,args)
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    restore(saver,sess,model_dir=args.model_dir, model_prefix=args.algo)
    logger.info('Predicting answers for test set...')
    test_batches = brc_data.gen_mini_batches('test', args.batch_size,
                                             pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    evaluate(args, test_batches, vocab, sess, result_dir=args.result_dir,result_prefix="test.predicted")
def find_best_answer(sample, start_prob, end_prob, padded_p_len, args,para_prior_scores=None):
        """
        Finds the best answer for a sample given start_prob and end_prob for each position.
        This will call find_best_answer_for_passage because there are multiple passages in a sample
        """
        best_p_idx, best_span, best_score = None, None, 0
        for p_idx, passage in enumerate(sample['documents']):
            if p_idx >= args.max_p_num:
                continue
            passage_len = min(args.max_p_len, len(passage['segmented_passage']))
            answer_span, score = find_best_answer_for_passage(
                start_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                end_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                args,
                passage_len)
            if para_prior_scores is not None:
                # the Nth prior score = the Number of training samples whose gold answer comes
                #  from the Nth paragraph / the number of the training samples
                score *= para_prior_scores[p_idx]
            if score > best_score:
                best_score = score
                best_p_idx = p_idx
                best_span = answer_span
        if best_p_idx is None or best_span is None:
            best_answer = ''
            segmented_pred = []
        else:
            segmented_pred = sample['documents'][best_p_idx]['segmented_passage'][best_span[0]: best_span[1] + 1]
            best_answer = ''.join(segmented_pred)
        return best_answer, segmented_pred

def find_best_answer_for_passage(start_probs,end_probs,args,passage_len=None):
        """
        Finds the best answer with the maximum start_prob * end_prob from a single passage
        """
        if passage_len is None:
            passage_len = len(start_probs)
        else:
            passage_len = min(len(start_probs), passage_len)
        best_start, best_end, max_prob = -1, -1, 0
        for start_idx in range(passage_len):
            for ans_len in range(args.max_a_len):
                end_idx = start_idx + ans_len
                if end_idx >= passage_len:
                    continue
                prob = start_probs[start_idx] * end_probs[end_idx]
                if prob > max_prob:
                    best_start = start_idx
                    best_end = end_idx
                    max_prob = prob
        return (best_start, best_end), max_prob
def save(saver,sess,model_dir,model_prefix="model.ckpt"):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        logger = logging.getLogger("brc")
        saver.save(sess, os.path.join(model_dir, model_prefix))
        logger.info('Model saved in {}, with prefix {}.'.format(model_dir, model_prefix))

def restore(saver,sess,model_dir, model_prefix="model.ckpt"):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        logger = logging.getLogger("brc")
        saver.restore(sess, r""+os.path.join(model_dir, model_prefix))
        logger.info('Model restored from {}, with prefix {}'.format(model_dir, model_prefix))
