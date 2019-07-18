from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W) # 这里输出C维向量。这里面的真值与非真值的导数还不一样
        # 真值由y[i]来索引
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                # 如果是真值，按真值的求导公式求它的偏导，偏导是一个D维向量，与X同
                # 公式见讲义
                full_scores = (scores - correct_class_score + 1) # C维向量，多算了一个delta
                full_scores[y[i]] -= 1 # 把多算的delta减掉
                dy_i = -X[i] * np.sum(np.where(full_scores<=0, 0, 1)) # 求得y_i的偏导
                dW[:, y[i]] += dy_i
                dW[:, y[i]] += reg * 2 * W[:, y[i]] # 正则项
            else:
                # 如果不是真值                
                margin = scores[j] - correct_class_score + 1 # note delta = 1
                if margin > 0:
                    loss += margin
                    dy_j = 1 * X[i]
                else:
                    dy_j = 0 * X[i]
                dW[:, j] += dy_j
                dW[:, j] += reg * 2 * W[:, j] # 正则项

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW /= num_train

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    indexes = np.arange(num_train)
    scores_NxC = X.dot(W) # 每一个样本的分值，按行堆成矩阵
    correct_Nx1 = scores_NxC[indexes, y].reshape(num_train, 1) # 选出所有的y_i
    scores_NxC = scores_NxC - correct_Nx1 + 1 # 针对y_i，多加了一个delta
    scores_NxC[indexes, y] -= 1 # 减去多加的delta
    loss = np.sum(np.where(scores_NxC < 0, 0, scores_NxC)) / num_train # max(0, x)的调用
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    margin_NxC = np.where(scores_NxC <= 0, 0, 1) 
    correct_Nx1 = -np.sum(margin_NxC, axis=1) # 计算真值的系数
    margin_NxC[indexes, y] = correct_Nx1 # 将真值的系数覆盖到原矩阵
    dW = X.T.dot(margin_NxC) / num_train # DxC
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
