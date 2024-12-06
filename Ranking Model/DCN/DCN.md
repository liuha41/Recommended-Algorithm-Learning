# DCN

Wide&Deep改进之作，使用Cross网络代替Wide部分

![image-20241206212512119](./assets/image-20241206212512119.png)

Cross网络的核心思想是以一种有效的方式应用显式特征交叉。交叉网络由交叉层组成，第l层交叉层输出为xl，第l+1层输出为：

<img src="./assets/image-20241206212600530.png" alt="image-20241206212600530" style="zoom: 67%;" />

wl、bl ∈ Rd是第l层的权重和偏置参数。每个交叉层在特征交叉f之后将其输入加回，并且映射函数f拟合xl+1 − xl的残差。

<img src="./assets/image-20241206212917514.png" alt="image-20241206212917514" style="zoom:67%;" />

