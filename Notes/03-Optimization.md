# Optimization
> Optimization is the process of finding the set of parameters W that minimize the loss function.

优化就是寻找一个最优的$W$，使得损失函数的值最小。

### 梯度下降
我们的目标是找一个$W$，使$loss$最小。讲义中介绍了三种方法：
+ 随机生成一个$W$，计算$loss$，把$loss$值最小的$W$保存下来
+ 随机生成一个$W$，每次迭代，生成一个随机方向$dW$，往那个方向移动一个步长。然后计算新的$W$的$loss$，如果得到改善，就把新的$W$保存下来
+ 在第二种的基础上，不进行随机下降，而是计算$W$的梯度，沿梯度下降
  
这里非常精彩地引出梯度下降，并且求导数还是用了极限，而不是公式法
$$
\frac{df(x)}{dx}=\mathop{lim} \limits_{h→0} \frac{f(x+h)−f(x)}{h}
$$
h取一个非常小的值，就可以求出$f$在$x$的导数。当$x$是一个向量时，导数不叫导数，而叫偏导数。所谓的梯度就是包含各个方向偏导的一个向量。

这种方法很好，不过有个问题，就是参数越多，计算越慢～

另一种计算梯度的方法是使用数学公式。用数学公式更快，可是更容易出错。所以，实践中会拿两种方法计算出来的梯度进行比较。这就叫梯度检查(gradient check)。

### 随机梯度下降

> Gradient from a mini-batch is a good approximation of the gradient of the full objective.

> Even though SGD technically refers to using a single example at a time to evaluate the gradient, you will hear people use the term SGD even when referring to mini-batch gradient descent

实践中，通过求一个batch的梯度来作为所有数据集梯度的近似，以便更加快速地更新$W$。batch是一个超参数，但通常不会用交叉验证来设置它的值，而是常常根据内存的多少而定。通常，我们设置它为2的幂，因为框架经常为对此优化计算速度。