# DeepFM

FNN把FM训练结果作为初始化权重，没有对神经网络结构进行调整，DeepFM将FM的模型结构与Wide&Deep模型进行融合，模型结构图如下所示：

<img src="./assets/image-20241209143334029.png" alt="image-20241209143334029" style="zoom: 67%;" />

用FM替换了原来的Wide部分，加强了浅层网络部分的特征组合能力，DeepFM模型的改进主要针对Wide部分不具备自动的特征组合能力