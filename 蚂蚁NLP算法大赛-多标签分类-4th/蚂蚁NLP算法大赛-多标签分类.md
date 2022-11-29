# 蚂蚁NLP算法大赛-多标签分类

此次竞赛为在蚂蚁集团实习时参与的内部算法赛，因此数据相关细节无法描述，总任务为文本的多标签分类。

#### 最终名次

4-th

## 任务描述

给定用户文本text，预测该文本对应的多个意图。

## 数据描述

数据量：30000

数据特点：样本不均衡与长尾问题；标签类别间存在混淆

## 参赛方案

基础模型：chinese-roberta-wwm-ext-large

尝试1：通过同义词替换实现数据增强，同义词采用[腾讯词向量](https://ai.tencent.com/ailab/nlp/en/embedding.html)

此次数据增强未能提分，但是多数情况下可以提分。

```python
def data_aug(text):
    aug_text = []
    text_new = list(jieba.cut(text))
    for j in range(len(text_new)):
        word = text_new[j]
        if word in wv_from_text:
            vec = wv_from_text[word]
            candi_list = wv_from_text.most_similar(positive = [vec], topn = 3)
            for candi in candi_list[1:]:
                if candi[1] < 0.9:
                    break
                new_text = copy.deepcopy(text_new)
                new_text[j] = candi[0]
                aug_text.append("".join(new_text))
    return aug_text
```

尝试2：输入文本的处理。经观察，部分label是文本中的子串，因此采用jieba将文本分词后，使用[SEP]拼接在后面。

此处理有一定的提升。

尝试3：加入FGM对抗训练，具有一定的提升。

尝试4：加入自定义学习率调整，具有一定提升。

尝试3和4再CCKS文档中有代码。

尝试5：更换基础模型，使用NEZHA系列模型，分数极低，f1掉了10%。（可能是使用有问题，后续没有继续尝试）

尝试6：使用focal loss，结果有一定的下降，说明此数据集的不平衡问题并不是主要问题。

尝试7：使用后四层特征，而不是直接使用CLS，分数有所下降。

尝试8：在现有数据集上继续预训练语言模型，有一定的提升。

尝试9：使用[BatchFormer](https://arxiv.org/abs/2203.01522)，有一定的提升，后续均在BatchFormer的基础上进行操作，模型如下：

```python
class Mymodel(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super(Mymodel, self).__init__()
        if "large" in model_name:
            self.linear_input_dim = 1024
        else:
            self.linear_input_dim = 768
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(self.linear_input_dim, num_labels)
        self.transformerEncoderLayer = torch.nn.TransformerEncoderLayer(d_model = self.linear_input_dim, nhead = 4, dim_feedforward = self.linear_input_dim, dropout=0.3)
    
    def forward(self, input_ids, token_type_ids, attention_mask, is_training):
        bert_output = self.bert(input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask).pooler_output
        trans_encoder_output = self.BatchFormer(bert_output, self.transformerEncoderLayer, is_training = is_training)
        classify_output_pre = self.classifier(bert_output)
        classify_output_aft = self.classifier(trans_encoder_output)
        if is_training:
            return classify_output_pre, classify_output_aft
        else:
            return classify_output_pre
        
    def BatchFormer(self, x, encoder, is_training):
        # in training, label = torch.cat([label,label],dim=0)
        if is_training:
            pre_x = x
            x = encoder(x.unsqueeze(1)).squeeze(1)
            x = torch.cat([pre_x, x],dim=0)
        return x
    def add_noise(self, x, perturb_noise = 0.05): # add U[-0.05, 0.05] noise to imporve model, self.add_noise(classify_output_pre)
        perturb = torch.empty_like(x).uniform_(-perturb_noise, perturb_noise)
        return x + perturb
```

训练时输出classify_output_pre, classify_output_aft两部分，经试验发现，只使用后一部分算交叉熵损失效果更好。

尝试10：BatchFormer + FGM不收敛，噪声调小后收敛。

尝试11：使用weightedsampler，采样概率分布为：（1）log(all_sample_num/label_count)；（2）1/label_num/label_count，效果提升均不明显

尝试12：使用[Rdrop](https://arxiv.org/abs/2106.14448)，但是没有收敛，后续放弃。但是论文中以及一些博客介绍是可以提分的。代码来自[官方GitHub](https://github.com/dropreg/R-Drop)

```python
import torch.nn.functional as F

# define your task model, which outputs the classifier logits
model = TaskModel()

def compute_kl_loss(self, p, q, pad_mask=None):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss

# keep dropout and forward twice
logits = model(x)

logits2 = model(x)

# cross entropy loss for classifier
ce_loss = 0.5 * (cross_entropy_loss(logits, label) + cross_entropy_loss(logits2, label))

kl_loss = compute_kl_loss(logits, logits2)

# carefully choose hyper-parameters
loss = ce_loss + α * kl_loss
```

尝试13：K折交叉验证，稳定上分，但是速度变慢，代码见CCKS总结。

尝试14：加入[LLD损失]([A Light Label Denoising Method with the Internal Data Guidance | OpenReview](https://openreview.net/forum?id=cfpUdeozplN))去噪，略微提升，但是超参敏感，容易不收敛。本次比赛实现较为简单，未和论文中拟合，直接使用Bert的pooler_output来计算LLD损失。

```python
# beta is set to 0.03, 0.05, 0.05, 0.1 and 0.2 for 0%, 10%, 20%, 30% and 40% noisy rate respectively.
beta = 0.05
T_sim = bert_output @ bert_output.t() / (torch.norm(bert_output,dim=1))**2
L_sim = classify_output_pre @ classify_output_pre.t() / (torch.norm(classify_output_pre,dim=1))**2
lld_loss = ((T_sim > beta) * (1 - L_sim)**2).mean()
```

尝试15：融合多个模型，稳定上分。融合方式为多个模型分别训练，对测试数据的logits相加。

尝试16：使用attention融合BERT所有层的特征，效果较差。

尝试17：使用[p-tuning](https://arxiv.org/abs/2110.07602)，效果不好。

尝试18：只采用多模型融合，不采用交叉验证，去掉部分dropout，效果有一定提升。

## 总结

稳定上分的Trick：

- 对输入文本进行个性化处理，即根据任务处理文本。
- 若无数据泄露，即需要模型的鲁棒性而不是过拟合，加入对抗训练会有提升。
- 训练过程动态学习率的调整，会有一定提升。
- 在比赛数据集中进行继续预训练。
- 使用BatchFormer（不敢保证
- 多模型融合与多折交叉验证模型的融合。