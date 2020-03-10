# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements data process strategies.
"""

import os
import json
import logging
import numpy as np

class BRCDataset(object):
    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    """
    def __init__(self, max_p_num, max_p_len, max_q_len,gpus,batch_size,
                 train_files=[], dev_files=[], test_files=[]):
        self.logger = logging.getLogger("brc")
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len
        self.train_answer_len_cut_bins = 0
        self.train_set, self.dev_set, self.test_set = [], [], []
        if train_files:
            for train_file in train_files:
                self.train_set += self._load_dataset(train_file, train=True)

        if dev_files:
            for dev_file in dev_files:
                self.dev_set += self._load_dataset(dev_file)

        if test_files:
            for test_file in test_files:
                self.test_set += self._load_dataset(test_file)
        self._batch_for_gpus(len(gpus), batch_size)
        self.logger.info('Train set size: {} questions.'.format(len(self.train_set)))
        self.logger.info('Dev set size: {} questions.'.format(len(self.dev_set)))
        self.logger.info('Test set size: {} questions.'.format(len(self.test_set)))
    def _batch_for_gpus(self,num_gpu,batch_size):
        train_sample_num=len(self.train_set)
        device_batchs=train_sample_num // (num_gpu*batch_size)
        real_sample_num = device_batchs * num_gpu*batch_size
        remain_num = train_sample_num - real_sample_num
        if remain_num <= 0:
            return 
        if (remain_num // batch_size) >= (num_gpu // 2):
            append_num = num_gpu*batch_size - remain_num
            append_sample=self.dev_set[:append_num]
            self.dev_set = self.dev_set[append_num:]
            self.train_set += append_sample
        else:
            cut_sample = self.train_set[-remain_num:]
            self.train_set = self.train_set[:-remain_num]
            self.dev_set += cut_sample
    def _load_dataset(self, data_path, train=False):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
            [1,2,3,4,5,2]

        """
        with open(data_path,"r",encoding="utf-8") as fin:
            data_set = []
            for _, line in enumerate(fin):
                sample = json.loads(line.rstrip())
                if train:
                    if len(sample['segmented_question']) == 0 or len(sample['documents']) == 0:
                        continue
                    if len(sample['fake_answers']) == 0:
                        continue
                    if 'answer_docs' in sample:
                        sample['answer_passages'] = sample['answer_docs']
                    best_match_doc_ids = []
                    best_match_scores = []
                    answer_labels = []
                    fake_answers = []
                    for ans_idx, answer_label in enumerate(sample['answer_labels']):
                        # 对于 multi-answer 有的fake answer 没有找到
                        if answer_label[0] == -1 or answer_label[1] == -1:
                            continue
                        best_match_doc_ids.append(sample['best_match_doc_ids'][ans_idx])
                        best_match_scores.append(sample['best_match_scores'][ans_idx])
                        answer_labels.append(sample['answer_labels'][ans_idx])
                        fake_answers.append(sample['fake_answers'][ans_idx])
#                         print("fake_answer",sample['fake_answers'][ans_idx])
#                         print("source_answer",segment_passage)
                    if len(best_match_doc_ids) == 0:
                        continue
                    else:
                        sample['best_match_doc_ids'] = best_match_doc_ids
                        sample['answer_labels'] = answer_labels
                        sample['best_match_scores'] = best_match_scores
                        sample['fake_answers'] = fake_answers
                data_set.append(sample)
        return data_set

    def word_iter(self, set_name=None):
        """
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name == 'train' and self.train_answer_len_cut_bins > 0:  # 存在 bin
            data_set = []
            for bin_set in self.bin_cut_train_sets:
                data_set += bin_set
        elif set_name == 'train' and self.train_answer_len_cut_bins <= 0:  # 不存在 bin
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))

        if data_set is not None:
            for sample in data_set:
                for token in sample['segmented_question']:
                    yield token
                for doc in sample['documents']:
                    for token in doc['segmented_passage']:
                        yield token

    def convert_to_ids(self, vocab, use_oov2unk=False):
        """
        Convert the question and passage in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
            use_oov2unk: 所有oov的词是否映射到 <unk>, 默认为 False
        """
        # 如果是train, 则丢弃segmented_passage字段
        if self.train_answer_len_cut_bins > 0:  # 存在 bin
            for bin_set in self.bin_cut_train_sets:
                for sample in bin_set:
                    sample['question_token_ids'] = vocab.convert_to_ids(sample['segmented_question'], use_oov2unk)
                    for doc in sample['documents']:
                        doc['passage_token_ids'] = vocab.convert_to_ids(doc['segmented_passage'], use_oov2unk)
                        doc['segmented_passage'] = []
        elif self.train_set:
            for sample in self.train_set:
                sample['question_token_ids'] = vocab.convert_to_ids(sample['segmented_question'], use_oov2unk)
                for doc in sample['documents']:
                    doc['passage_token_ids'] = vocab.convert_to_ids(doc['segmented_passage'], use_oov2unk)

        for data_set in [self.dev_set, self.test_set]:
            if data_set is None:
                continue
            for sample in data_set:
                sample['question_token_ids'] = vocab.convert_to_ids(sample['segmented_question'], use_oov2unk)
                for doc in sample['documents']:
                    doc['passage_token_ids'] = vocab.convert_to_ids(doc['segmented_passage'], use_oov2unk)

    def get_data_length(self, set_name):
        if set_name == 'train' and self.train_answer_len_cut_bins > 0:
            return sum([len(bin_set) for bin_set in self.bin_cut_train_sets])
        elif set_name == 'train' and self.train_answer_len_cut_bins <= 0:
            return len(self.train_set)
        elif set_name == 'dev':
            return len(self.dev_set)
        elif set_name == 'test':
            return len(self.test_set)
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))

    def gen_mini_batches(self, set_name, batch_size, pad_id, shuffle=True):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            shuffle: if set to be true, the data is shuffled.
            calc_total_batch_cnt: used to calc the total batch counts
        Returns:
            a generator for all batches
        """
        if set_name == 'train':
            data_set = self.train_set
            is_testing = False
        elif set_name == 'dev':
            data_set = self.dev_set
            is_testing = False
        elif set_name == 'test':
            data_set = self.test_set
            is_testing = True
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))

        data_size = len(data_set)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data_set, batch_indices, pad_id, is_testing=is_testing)

    def _one_mini_batch(self, data, indices, pad_id, is_testing):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        batch_raw_data = []
        for i in indices:
            sample = data[i]
            cleaned_sample = {'documents': [{'segmented_passage': doc['segmented_passage']} for doc in sample['documents']],
                              'question_id': sample['question_id'],
                              'question_type': sample['question_type'],
                              'segmented_question': sample['segmented_question']}
            if 'segmented_answers' in sample:
                cleaned_sample['segmented_answers'] = sample['segmented_answers']
            if 'answer_labels' in sample:
                cleaned_sample['answer_labels'] = sample['answer_labels']

            batch_raw_data.append(cleaned_sample)

        batch_data = {'raw_data': batch_raw_data,
                      'question_token_ids': [],
                      'question_length': [],
                      'passage_token_ids': [],
                      'passage_length': [],
                      'start_ids': [],
                      'end_ids': [],
                      'match_scores': []}

        batch_samples = [data[i] for i in indices]

        max_passage_num = max([len(sample['documents']) for sample in batch_samples])
        max_passage_num = min(self.max_p_num, max_passage_num)
        # 增加信息,求最大答案数
        if not is_testing:
            max_ans_num = max([len(sample['answer_labels']) for sample in batch_samples])
        else:
            max_ans_num = 1

        for _, sample in enumerate(batch_samples):
            for pidx in range(max_passage_num):
                if pidx < len(sample['documents']):
                    batch_data['question_token_ids'].append(sample['question_token_ids'])
                    batch_data['question_length'].append(len(sample['question_token_ids']))
                    passage_token_ids = sample['documents'][pidx]['passage_token_ids']
                    batch_data['passage_token_ids'].append(passage_token_ids)
                    batch_data['passage_length'].append(min(len(passage_token_ids), self.max_p_len))
                else:
                    batch_data['question_token_ids'].append([])
                    batch_data['question_length'].append(0)
                    batch_data['passage_token_ids'].append([])
                    batch_data['passage_length'].append(0)
        batch_data, padded_p_len, _ = self._dynamic_padding(batch_data, pad_id)

        if not is_testing:
            for sample in batch_samples:
                start_ids = []
                end_ids = []
                scores = []
                for aidx in range(max_ans_num):
                    if aidx < len(sample['best_match_doc_ids']):
                        gold_passage_offset = padded_p_len * sample['best_match_doc_ids'][aidx]
                        start_ids.append(gold_passage_offset + sample['answer_labels'][aidx][0])
                        end_ids.append(gold_passage_offset + sample['answer_labels'][aidx][1])
                        scores.append(sample['best_match_scores'][aidx])
                    else:
                        start_ids.append(0)
                        end_ids.append(0)
                        scores.append(0)
                batch_data['start_ids'].append(start_ids)
                batch_data['end_ids'].append(end_ids)
                batch_data['match_scores'].append(scores)
        else:
            # test阶段
            batch_data['start_ids'] = [[0] * max_ans_num] * len(indices)
            batch_data['end_ids'] = [[0] * max_ans_num] * len(indices)
            batch_data['match_scores'] = [[0] * max_ans_num] * len(indices)
        return batch_data

    def _dynamic_padding(self, batch_data, pad_id):
        """
        Dynamically pads the batch_data with pad_id
        """
        pad_p_len = min(self.max_p_len, max(batch_data['passage_length']))
        pad_q_len = min(self.max_q_len, max(batch_data['question_length']))
        batch_data['passage_token_ids'] = [
            (ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len] for ids in batch_data['passage_token_ids']]
        batch_data['question_token_ids'] = [
            (ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len] for ids in batch_data['question_token_ids']]
        return batch_data, pad_p_len, pad_q_len
