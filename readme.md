### 介绍
>> 非estimator GPU并行和梯度累计实现
>> 实验模型多文档机器阅读理解模型  
>> 数据集百度Dureader  

### 要求
 tensorflow>=1.10

### 实验结果
实验数据为小数据实验：1000条训练集，100条验证集，数据集原因结果会有波动
|参数 |  rouge-l |bleu4 |
|  ----  | ----  |----  |
| batch-size:8; gpu-num:1 | 41. 2759| 40.4671 |
| batch-size 2;gpu-num:4 |40.7620|39.8543 |
| batch-size 1;gpu-num:1 accumulate_step:8 | 42.0745| 40.7764 |

### 说明
这里只提供GPU并行和梯度累计代码，模型代码暂不提供