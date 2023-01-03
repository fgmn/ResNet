# ResNet

使用ResNet网络进行十种食物图像分类，基于迁移学习方法训练

### Quick Start

1. 更改环境文件``freeze.yml``中``prefix``路径，运行

   ```shell
   conda env create -f freeze.yml
   ```

   快速创建环境；

2. 进入``food_data``目录下运行``split.py``，将数据集划分为训练集和测试集；

3. 训练网络，运行``train.py``，脚本将在验证集上表现最优的网络``resNet34.pth``保存在代码目录下；

4. 单张图片预测，运行``predict.py``，注意修改图片路径；

5. 测试集预测，运行``batch_predict.py``，将生成``result.txt``包含对500张图片的预测结果。
