#coding=utf-8
'''
Created on 2020年3月8日
@author: Administrator
@email: 1113471782@qq.com
'''
import tensorflow as tf
import collections
import numpy as np
from tensorflow.python.framework.ops import IndexedSlicesValue
class AccumulateSteps(object):
    def __init__(self,grads_vars,accumulate_step=2):
        """
            grads_vars:[(g1,v1),(g2,v2)]
        """
        assert accumulate_step >0
        self.grads_holder=[]
        self.grads_accumulator=collections.OrderedDict()
        self.local_step=0
        self.accumulate_step=accumulate_step
        self.IndexedSlices_index=None
        self.IndexedSlices_value=None
        self.IndexedSlices_Dense_shape=None
        for (g, v) in grads_vars:
            if g is None: continue
            if isinstance(g, tf.IndexedSlices):
                self.IndexedSlices_index = tf.placeholder(dtype=tf.int32,shape=g.indices.get_shape())
                self.IndexedSlices_value = tf.placeholder(dtype=g.values.dtype,shape=g.values.get_shape())
                self.IndexedSlices_Dense_shape = tf.placeholder(dtype=g.dense_shape.dtype, shape=g.dense_shape.get_shape())
                grade_IndexedSlices=tf.IndexedSlices(self.IndexedSlices_value,
                                                     self.IndexedSlices_index,
                                                     dense_shape=self.IndexedSlices_Dense_shape)
                self.grads_holder.append((grade_IndexedSlices,v))
            else:
                self.grads_holder.append((tf.placeholder(dtype=g.dtype, shape=g.get_shape()), v))
    def set_local_step(self,step):
        self.local_step =step
    def _generate_grads_dict(self):
        feed_dict={}
        feed_dict.update({self.IndexedSlices_index:self.grads_accumulator["indices"],
                            self.IndexedSlices_value:self.grads_accumulator["values"],
                            self.IndexedSlices_Dense_shape:self.grads_accumulator["dense_shape"]})
        for holder_index,placeholder in enumerate(self.grads_holder):
            if holder_index <=0:continue
            feed_dict.update({placeholder[0]:self.grads_accumulator[holder_index]})
        self.grads_accumulator.clear()
        self.local_step=0
        return feed_dict
    def add_grads(self,grad_vars,right_grads=False):
        if right_grads:
            return self._generate_grads_dict()
        self.local_step +=1
        assert len(grad_vars) == len(self.grads_holder)
        for g_uid,(g,v) in enumerate(grad_vars):
            if isinstance(g,IndexedSlicesValue):
                if "indices" in self.grads_accumulator:
                    self.grads_accumulator["indices"] = np.concatenate((self.grads_accumulator["indices"],g.indices),axis=0)
                else:
                    self.grads_accumulator["indices"] = g.indices
                if "values" in self.grads_accumulator:
                    self.grads_accumulator["values"] = np.concatenate((self.grads_accumulator["values"],g.values),axis=0)
                else:
                    self.grads_accumulator["values"] = g.values
                self.grads_accumulator["dense_shape"] = g.dense_shape
            else:
                if g_uid in self.grads_accumulator:
                    self.grads_accumulator[g_uid]=sum([self.grads_accumulator[g_uid],g])
                else:
                    self.grads_accumulator[g_uid]=g
        
        if self.local_step == self.accumulate_step:
            
            return self._generate_grads_dict()
        else:
            return None
        
        
        
        
                
