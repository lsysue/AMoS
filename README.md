# AMoS: Autunomous Multimodal Standardization of Point of Interest for Location-Based Services

## Code Tree:

```
AMoS
|- analysis/                // 用于数据分析的notebook
|- checkpoints/             // 保存训练的中间结果、模型参数以及测试结果，按照train的时间分建文件夹
|- config/                  // 配置参数
|- misc/                    // 分布式训练相关函数
|- data/                    // 原始数据及预处理（经过Chinesebert, gpsbert, mgeo等生成嵌入）的数据
|- datasets/               // 构造数据集
|- clustering/              // 聚类算法
|- models/                  // 组件models
    |- chinesebert/
    |- gpsbert/
    |- mgeo/
    |- transformer/
    |- ggnn/
|- loss/                    // 损失函数
|- train_classification.py  // 将聚类结果或真值标签监督图神经网络训练后对边关系进行二分类，loss为CEloss
|- train_clustering.py      // 将聚类结果或真值标签监督图神经网络训练后生成节点嵌入，对节点嵌入进行聚类，loss为对比学习loss
```

## 数据预处理

### 数据字段整理和重命名

POI数据以及Wi-Fi数据相关处理代码在 `./data/data_process.ipynb`

### 相关模型预训练或微调

**Chinesebert**：官方repo--[https://github.com/ShannonAI/ChineseBert]。
没有重新做pretrain，用已有模型参数 `AMoS/models/chinesebert/chinbert-base/`

**GPSBert**：

1. 准备数据：包含*骑手配送轨迹*的订单记录，放在data文件夹下。Note：订单时间跨度越长越好，16天数据训练出来效果不佳。
2. 准备预训练词库和语料库 -> geo_vocab.txt & traj_corpus.txt。Note：可自行调整代码中的PRECISION，即经纬度的保留精度
   ```
   python gen_data.py
   ```
3. 开始预训练，*注意修改VOCAB_SIZE*
   ```
   python pretrain.py
   ```

**MGeo**：官方repo--[https://github.com/PhantomGrapes/MGeo]。

1. 准备数据：依照data/train_dataset.jsonl格式准备训练数据集，依照configs/Rerank.yaml格式准备训练参数文件
2. 预训练组件models
   ```
   bash run_gis_encoder_pretrain.sh
   bash run_mm_pretrain.sh
   ```
3. 将 `Mgeo/output/Pretrain_mm/checkpoint_best.pth`copy至本repo `./models/mgeo/`目录下。

### 生成文本与地理初始嵌入

Note: 该代码使用的配置参数为 `./config/default_params.py`。

```
cd ./datasets/
python eleme.py # 注意修改其中的data_path
```

## 训练及测试

train scripts:

```
CUDA_VISIBLE_DEVICES=3 python train_graph.py --cfg=./config/custom_configs/{train_config.yaml} --data_path={processed_data_path} --epoch=1000
```

```
CUDA_VISIBLE_DEVICES=3 python train_classification.py --cfg=./config/custom_configs/{train_config.yaml} --data_path={processed_data_path} --epoch=500
```

```
CUDA_VISIBLE_DEVICES=3 python train_clustering.py --cfg=./config/custom_configs/{train_config.yaml} --data_path={processed_data_path} --epoch=500
```

evaluate script:

```
python evaluate.py --cfg=./config/custom_configs/{evaluate_config.yaml} --data_path={processed_data_path} --checkpoint_path=./checkpoints/False_8-20-15-42/basenet-4/checkpoint_60.pth.tar
```