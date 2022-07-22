# Jittor 可微渲染新视角生成比赛 RefNGP
## 简介
本项目包含了第二届计图挑战赛计图 -可微渲染新视角生成比赛的代码实现。没有实质性创新成果，涨分点主要在调参以及没有实现完的Ref-NeRF. 最终B榜排名11.

![image-20220722215100409](https://s2.loli.net/2022/07/22/l9zM14n5JL3Cjte.png)

## 安装 
本项目可在 1张Tesla T4上运行，单个场景执行80000次训练，耗时约30分钟。

#### 运行环境
- ubuntu 20.04 LTS
- python >= 3.7
- jittor >= 1.3.0

#### 安装依赖
执行以下命令安装 python 依赖
```
pip install -r requirements.txt
```

#### 训练数据

请移步[jrender/download_competition_data.sh at main · Jittor/jrender (github.com)](https://github.com/Jittor/jrender/blob/main/download_competition_data.sh).

#### 预训练模型

预训练模型很大（1.9G左右），请移步此处下载。

## 训练
可以直接运行以下命令
```bash
bash run.sh
```

如果要执行单场景训练，可以运行

```bash
python tools/run_net.py --config-file xxx.py
```

## 推理

生成比赛结果可以使用以下命令

```
python test.py
```

进行单个场景推理，可以使用

```bash
python tools/run_net.py --config-file xxx.py --task test
```

xxx.py中要指明ckpt_path

## 致谢

此项目基于[Jittor/JNeRF: JNeRF is a NeRF benchmark based on Jittor. JNeRF re-implemented instant-ngp and achieved same performance with original paper. (github.com)](https://github.com/Jittor/JNeRF)实现，并参考了论文*[Ref-nerf: Structured view-dependent appearance for neural radiance fields](https://arxiv.org/abs/2112.03907)*.

