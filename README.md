# Cross-directional Center Network (CCNet) and MSVR310
Paper: Multi-spectral Vehicle Re-identification with Cross-directional  Center Network and A High-quality Benchmark

Dataset: ##link: https://pan.baidu.com/s/1QyZUkbvpZ3U4d0iPt4IfVA ##code:msvr

本代码整体框架适用的是罗浩博士的 Bag of Tricks的代码框架
论文：A Strong Baseline and Batch Normalization Neck for Deep Person Re-Identification
但是由于使用时间较早，本代码种的内容与现在github上的内容存在一定出入，如果需要修改请一定注意（下文给出了本代码具体包和版本）。
github: https://github.com/michuanhaohao/reid-strong-baseline
请务必注意pytorch-ignite包的版本，本实验为 0.2.0，下有其他包的版本信息

文件结构：
config：
    default.py :包含所有命令行参数默认配置，yml文件种不包含的则按此处默认值，如需要添加参数，先在此处添加。
data：
    datasets: 包含所有数据集的读取方式，获取初始图片路径，标签等（重要）
    sampler：三元组采样过程代码
    transforms：数据预处理部分
    build.py: 构建训练和测试数据的loader，指定数据读取后的处理方式
    collate_batch.py: 具体的数据读取后处理方式
engine：
    inference.py: 测试时网络运算过程代码（重要）
    trainer***.py: 训练时网络运算过程代码，训练的核心代码在这里（重要）
layers:
    工具中一些损失函数和我自己写的一些早期尝试，可以对比github上的看，多出来的就是我瞎写的
modeling:
    各个网络模型
    backbones: 骨干网络代码
    baseline_zxp.py: CCNet结构代码 （重要）
modeling_fastreid:
    移植了HRCN到本工具中
outputs：
    输出文件
solver：
    优化器相关
utils：一些工具和度量代码
    reid_metirc.py :此代码负责测试时具体指标计算 （重要）
v: 一些可视化代码
    这里都是我自己的一些可视化或者统计分析代码，如果要学习tSNE、grad-CAM等可视化工具，建议先自己搜索相关原始代码
    我经常喜欢在分析前，模型加载参数把数据集跑一遍测试，然后把所有特征保存起来，这样后面分析方便，直接加载保存的文件即可。

clearSpace.py: 清理指定文件夹下的文件（不重要）

dark_enhance.py: 网上找的利用去雾算法实现暗光增强，我有改动（不重要）

feat_visualize.py: 部分特征可视化代码

save_feat.py: 获取指定模型的特征

test.py: 测试入口（重要）

train.py: 训练入口（重要）

resultAna.py: 根据log文件分析loss，精度的代码

run_list.py: 平时连续跑实验，代码无改动，但是手动指定的参数有改动时，用这个省事

wait2run.py: 服务器上抢GPU用的，当年也实在抢不到卡才这么干的

***.yml: 训练配置文件，我以softmax_triplet_注释版.yml为例，在里面写了详细注释，使用时请不要直接使用注释版，注释版只是注释，不能直接用

re,rank**.txt: 测试时gallery样本对应每个query的排序结果

另外，在编写代码和配置文件时，应该尽量少用中文，因为部分包例如cv2，yacs等对中文支持存在缺陷。

训练的默认配置文件是：softmax_triplet.yml，需要更换可以用 --config_file=xxx.yml更换

训练指令实例：
    第一个指令指定了测试时保存文件夹的名字 和 损失函数的权重
    python train.py NAME "CdC_alpha06lam07" ALPHA 0.6 LAMBDA 0.7
    第二个指令指定了测试时保存文件夹的名字 和 最大训练epoch
    python train.py NAME "CdC_alpha06lam04" SOLVER.MAX_EPOCHS 1000
    第三个指令指定了训练配置文件，其他和第二个一样
    python train.py  --config_file=rnt100.yml NAME "CdC_alpha06lam04" SOLVER.MAX_EPOCHS 1000
