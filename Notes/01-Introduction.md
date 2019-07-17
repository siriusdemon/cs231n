# Introduction

### Challenges
+ Viewpoint variation. A single instance of an object can be oriented in many ways with respect to the camera.
+ Scale variation. Visual classes often exhibit variation in their size (size in the real world, not only in terms of their extent in the image).
+ Deformation. Many objects of interest are not rigid bodies and can be deformed in extreme ways.
+ Occlusion. The objects of interest can be occluded. Sometimes only a small portion of an object (as little as few pixels) could be visible.
+ Illumination conditions. The effects of illumination are drastic on the pixel level.
+ Background clutter. The objects of interest may blend into their environment, making them hard to identify.
+ Intra-class variation. The classes of interest can often be relatively broad, such as chair. There are many different types of these objects, each with their own appearance.

### Nearest Neighbor Classifier
最近邻算法

+ 训练过程不计算，只是保存所有训练样本
+ 测试过程，将测试样本与所有训练样本进行比对，计算一个距离值
+ 选出距离最小的样本，作为测试样本的预测结果
+ k-近邻算法，则是取出距离最近的k的样本，选其中的众数标签

#### L2距离
$$
d_2(I_1,I_2)=\sqrt{∑\limits_p(I^p_1−I^p_2)^2}
$$

L2距离跟L1距离相比有什么特点呢？给定两个向量来比较，L2距离更偏好两个向量中的大部分值的差距比较接近，而不是某几个值差距很大，而其他的很小。

### 交叉验证
> In cases where the size of your training data (and therefore also the validation data) might be small, people sometimes use a more sophisticated technique for hyperparameter tuning called cross-validation. Working with our previous example, the idea is that instead of arbitrarily picking the first 1000 datapoints to be the validation set and rest training set, you can get a better and less noisy estimate of how well a certain value of k works by iterating over different validation sets and averaging the performance across these. For example, in 5-fold cross-validation, we would split the training data into 5 equal folds, use 4 of them for training, and 1 for validation. We would then iterate over which fold is the validation fold, evaluate the performance, and finally average the performance across the different folds.

交叉验证就是用来调超参的一种方法，不过就是计算量很大就是了。