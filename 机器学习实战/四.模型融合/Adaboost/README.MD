# 一、简介

这个方法主要涉及到2个权重集合：

**样本的权重集合**
每个样本都对应一个权重。 在构建第一个弱模型之前，所有的训练样本的权重是一样的。第一个模型完成后，要加大那些被这个模型错误分类(分类问题)、或者说预测真实差值较大(回归问题)的样本的权重。依次迭代，最终构建多个弱模型。每个弱模型所对应的训练数据集样本是一样的，只是数据集中的样本权重是不一样的。
**弱模型的权重集合**
得到的每个弱模型都对应一个权重。精度越高(分类问题的错分率越低，回归问题的错误率越低)的模型，其权重也就越大，在最终集成结果时，其话语权也就越大。

![adaboost流程图](https://github.com/nxf75/ML_Library/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%AE%9E%E6%88%98/%E5%9B%9B.%E6%A8%A1%E5%9E%8B%E8%9E%8D%E5%90%88/Adaboost/pic/Adaboost%E6%B5%81%E7%A8%8B%E5%9B%BE.png)

 

# 二、步骤

## **1.分类问题**

- **训练数据集**

　　　![img](https://img2018.cnblogs.com/blog/1252882/201906/1252882-20190614151124475-231257415.png)

　　　令Yi = 1 or -1，这种定义便于后面的结果集成。集合Y0表示数据集样本的真实类别序列。

- **初始的样本权重集合S1，弱模型的权重集合为D**　　

　　   ![img](https://img2018.cnblogs.com/blog/1252882/201906/1252882-20190614151353549-2097300745.png)

　　   ![img](https://img2018.cnblogs.com/blog/1252882/201906/1252882-20190614151425617-1855567669.png)

　　  n为数据集样本个数，m为要建立的弱模型的个数

- **针对数据集构建弱模型M1，得到这个弱模型的错分率为**

　　　假设弱模型M1的训练数据集的预测类别序列为P1，预测数据集的预测类别序列为Pre_1。

​          [![err=\frac{C_{error}}{C_{Data}}=\frac{\left \| P1_{i} \neq Y0_{i},i=1,2\cdots ,n \right \|}{n}](https://camo.githubusercontent.com/62d6ee7a14d935b398360f5e4c5918abcdde3153/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f6572723d5c667261637b435f7b6572726f727d7d7b435f7b446174617d7d3d5c667261637b5c6c6566742673706163653b5c7c2673706163653b50315f7b697d2673706163653b5c6e65712673706163653b59305f7b697d2c693d312c325c63646f74732673706163653b2c6e2673706163653b5c72696768742673706163653b5c7c7d7b6e7d)](https://www.codecogs.com/eqnedit.php?latex=err=\frac{C_{error}}{C_{Data}}=\frac{\left&space;\|&space;P1_{i}&space;\neq&space;Y0_{i},i=1,2\cdots&space;,n&space;\right&space;\|}{n})，

　　   其中Cerror表示被弱模型M1错分的样本个数，CData为全部的样本个数，也就是n。

- **计算弱模型M1的权重**

　　  ![img](https://img2018.cnblogs.com/blog/1252882/201906/1252882-20190614151708179-1094486470.png)

- **更改样本的权重**

　　  ![img](https://img2018.cnblogs.com/blog/1252882/201906/1252882-20190614151736569-1239150406.png)

　　  D1为非负数，因此预测正确的样本的权重会比上一次的降低，预测错误的会比上一次的增高。

　　  其中除以sum(S)，相当于将样本权重规范化。

- **迭代**

　　  当达到设定的迭代次数时停止，或者错分率小于某个小的正数时停止迭代。

　　  此时得到m个弱模型，以及预测数据集对应的预测结果序列Pre_1，Pre_2， ……Pre_m，

　　  以及模型的权重集合D。

- **结果集成**

　　 针对第i个预测样本的集成结果为JI_i,

​        [![\large JI\_i = \mathbf{sign}\sum_{k=1}^{m}Dk*Pre\_k_i](https://camo.githubusercontent.com/26d506e5bacaa76feaba1dfc5461ad3ab045e03f/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f5c6c617267652673706163653b4a495c5f692673706163653b3d2673706163653b5c6d61746862667b7369676e7d5c73756d5f7b6b3d317d5e7b6d7d446b2a5072655c5f6b5f69)](https://www.codecogs.com/eqnedit.php?latex=\large&space;JI\_i&space;=&space;\mathbf{sign}\sum_{k=1}^{m}Dk*Pre\_k_i)，sign为符号函数。

　　 

　　回归问题和分类问题的最大不同在于，回归问题错误率的计算不同于分类问题的错分率，下面给出回归问题的步骤，因为回归算法有很多的变种，这里以Adaboost R2算法为例说明：

 

## **2.回归问题**

-  **训练数据集**

​          [![Data=\{(X1,Y1),(X2,Y2),\cdots,(Xn,Yn)\}](https://camo.githubusercontent.com/97e1f68acba97319518144ff7f9aec18b22b079f/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f446174613d5c7b2858312c5931292c2858322c5932292c5c63646f74732c28586e2c596e295c7d)](https://www.codecogs.com/eqnedit.php?latex=Data=\{(X1,Y1),(X2,Y2),\cdots,(Xn,Yn)\})，输出值的序列为Y0。

- **初始的样本权重集合S1，弱模型的权重集合为D**

​         ![img](https://img2018.cnblogs.com/blog/1252882/201906/1252882-20190614152303368-1879420379.png)

​          ![img](https://img2018.cnblogs.com/blog/1252882/201906/1252882-20190614152321845-108318760.png)

　　  n为数据集样本个数，m为要建立的弱模型的个数

- **针对数据集构建弱模型M1，得到这个弱模型的错误率为**

    假设弱模型M1的训练数据集的预测类别序列为P1，预测数据集的预测类别序列为Pre_1。

    [![\large maxerr = \mathbf{max} (|Y0i-P1i|,i=1,2,\cdots,n)](https://camo.githubusercontent.com/04617a6158ac0d8f0818f9ce2b8a906f6463aa77/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f5c6c617267652673706163653b6d61786572722673706163653b3d2673706163653b5c6d61746862667b6d61787d2673706163653b287c5930692d5031697c2c693d312c322c5c63646f74732c6e29)](https://www.codecogs.com/eqnedit.php?latex=\large&space;maxerr&space;=&space;\mathbf{max}&space;(|Y0i-P1i|,i=1,2,\cdots,n))

    - **误差损失为线性**

        [![\large err_i=\frac{|Y0i-P1i|}{maxerr}](https://camo.githubusercontent.com/133ddaab1793739aff8bd230c65144f45bbab0be/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f5c6c617267652673706163653b6572725f693d5c667261637b7c5930692d5031697c7d7b6d61786572727d)](https://www.codecogs.com/eqnedit.php?latex=\large&space;err_i=\frac{|Y0i-P1i|}{maxerr})

    - **误差损失为平方**

        [![\large erri=\frac{(Y0i-P1i)^{2}}{maxerr^{2}}](https://camo.githubusercontent.com/8079493fde17fbd40451b7692e82f26e32251179/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f5c6c617267652673706163653b657272693d5c667261637b285930692d503169295e7b327d7d7b6d61786572725e7b327d7d)](https://www.codecogs.com/eqnedit.php?latex=\large&space;erri=\frac{(Y0i-P1i)^{2}}{maxerr^{2}})

    - **误差损失为指数**

        [![\large erri=1-e^{\frac{-|P1i-Y0i|}{maxerr}}](https://camo.githubusercontent.com/c40d8860b03c70f808585a6c6d780ab8a43117f4/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f5c6c617267652673706163653b657272693d312d655e7b5c667261637b2d7c5031692d5930697c7d7b6d61786572727d7d)](https://www.codecogs.com/eqnedit.php?latex=\large&space;erri=1-e^{\frac{-|P1i-Y0i|}{maxerr}})

    **错误率的计算公式为：**

    [![\large err=\sum_{i=1}^{N}S1i*erri](https://camo.githubusercontent.com/e7949e794e1cc07a9d391cf225cc951b4d936e00/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f5c6c617267652673706163653b6572723d5c73756d5f7b693d317d5e7b4e7d5331692a65727269)](https://www.codecogs.com/eqnedit.php?latex=\large&space;err=\sum_{i=1}^{N}S1i*erri)

- **计算弱模型M1的权重**

    [![\large D1=\mathbf{ln}\frac{1-err}{err}](https://camo.githubusercontent.com/d5d188747737f8908f6903ba09bee340247c2440/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f5c6c617267652673706163653b44313d5c6d61746862667b6c6e7d5c667261637b312d6572727d7b6572727d)](https://www.codecogs.com/eqnedit.php?latex=\large&space;D1=\mathbf{ln}\frac{1-err}{err})

- **更改样本的权重**

    [![\large \\S2i=\frac{S1i}{\mathbf{sum}(S)}*D1^{1-erri}\\ \mathbf{sum}(S)=\sum_{i=1}^{n}S1i*D1^{1-erri}](https://camo.githubusercontent.com/ae60ae56f8863115cc39ff20057020dc1830768d/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f5c6c617267652673706163653b5c5c5332693d5c667261637b5331697d7b5c6d61746862667b73756d7d2853297d2a44315e7b312d657272697d5c5c2673706163653b5c6d61746862667b73756d7d2853293d5c73756d5f7b693d317d5e7b6e7d5331692a44315e7b312d657272697d)](https://www.codecogs.com/eqnedit.php?latex=\large&space;\\S2i=\frac{S1i}{\mathbf{sum}(S)}*D1^{1-erri}\\&space;\mathbf{sum}(S)=\sum_{i=1}^{n}S1i*D1^{1-erri}) 。 其中除以sum(S)，相当于将样本权重归一化。

- **迭代**

    当达到设定的迭代次数时停止，或者错误率小于某个小的正数时停止迭代。

    此时得到m个弱模型，以及预测数据集对应的预测结果序列Pre_1，Pre_2， ……Pre_m，

    以及模型的权重集合D。

- **结果集成**

    针对第i个预测样本的集成结果为JI_i,

    [![\large JI\_i = \sum_{k=1}^{m} Dk*Pre\_k_i](https://camo.githubusercontent.com/a4b97f8e03a4e1b53b9b35b9becb9c4152c97ab1/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f5c6c617267652673706163653b4a495c5f692673706163653b3d2673706163653b5c73756d5f7b6b3d317d5e7b6d7d2673706163653b446b2a5072655c5f6b5f69)](https://www.codecogs.com/eqnedit.php?latex=\large&space;JI\_i&space;=&space;\sum_{k=1}^{m}&space;Dk*Pre\_k_i)

## 3.正则化

现在将回归问题和分类问题的最终的集成形式写为如下更为一般的形式

[![F_{m}(X)=\sum_{k=1}^{m}D_{k}*Pre_{k}(X)](https://camo.githubusercontent.com/a4bb153f8b7a734b214d62fc65858235c285a6fb/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f465f7b6d7d2858293d5c73756d5f7b6b3d317d5e7b6d7d445f7b6b7d2a5072655f7b6b7d285829)](https://www.codecogs.com/eqnedit.php?latex=F_{m}(X)=\sum_{k=1}^{m}D_{k}*Pre_{k}(X))

也就是有：[![F_{k}(X)= F_{k-1}(X) + D_{k}*Pre_{k}(X)](https://camo.githubusercontent.com/cecf20fce94f56f2d1945e75db67fe538fee62b2/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f465f7b6b7d2858293d2673706163653b465f7b6b2d317d2858292673706163653b2b2673706163653b445f7b6b7d2a5072655f7b6b7d285829)](https://www.codecogs.com/eqnedit.php?latex=F_{k}(X)=&space;F_{k-1}(X)&space;+&space;D_{k}*Pre_{k}(X))

现在将其正则化：

[![\\F_{k}(X)= F_{k-1}(X) + a *D_{k}*Pre_{k}(X)\\ F_{0}(X) = \overrightarrow{\mathbf{0}}](https://camo.githubusercontent.com/22f49bac2bdf994ac372db803f5bcb5c18212a4c/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f5c5c465f7b6b7d2858293d2673706163653b465f7b6b2d317d2858292673706163653b2b2673706163653b612673706163653b2a445f7b6b7d2a5072655f7b6b7d2858295c5c2673706163653b465f7b307d2858292673706163653b3d2673706163653b5c6f76657272696768746172726f777b5c6d61746862667b307d7d)](https://www.codecogs.com/eqnedit.php?latex=\\F_{k}(X)=&space;F_{k-1}(X)&space;+&space;a&space;*D_{k}*Pre_{k}(X)\\&space;F_{0}(X)&space;=&space;\overrightarrow{\mathbf{0}})

其中a为学习率，也就是步长。在调参时，和弱模型的最大个数一起调参。

 

 


 

# 三、疑点

- **增加的样本权重如何在下一个模型训练时体现出作用？**

　　答：以决策树为例，计算最佳分割点基尼系数时，或者MSE时，要乘以样本权重。总之，对于样本的计算量，要乘以相应的样本的权重。如果没在这个AdaBoost的框架下，则相当于原来的样本的权重都是一样的，所以乘或者不乘是一样的，现在在这个AdaBoost的框架下，因为样本权重发生了改变，所以需要乘。这样就把样本权重的改变体现了出来。

 

- **这个弱模型可以是什么？**

　　答：经常用的就是单层的决策树，也称为决策树桩(Decision Stump)，例如单层的CART。其实这个层数也是参数，需要交叉验证得到最好的。当然这个弱模型也可以是SVM、逻辑回归、神经网络等。

 

  
  
  
  
  
  


 

**参考文献：**

【1】[Machine-Learning-for-Beginner-by-Python3/Boosting/AdaBoost/](https://github.com/nxf75/Machine-Learning-for-Beginner-by-Python3/tree/master/Boosting/AdaBoost)
