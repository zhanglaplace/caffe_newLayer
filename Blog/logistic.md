---
title: Logistic回归分析
date: 2017-09-07 20:03:49
tags: [机器学习,统计学习方法]
categories: 机器学习
---
### Logistic回归分析
$\qquad Logistic回归为概率型非线性回归模型，机器学习常用的二分类分类器，其表达式为:$

$\quad \quad z=w_{1}\*x_{1}+w_{2}\*x_{2}+\cdots +w_{n}\*x_{n}+b=\sum_{i=0}^n w_{i}x_{i}  (其中 b等于w_{0}，x_{0}等于1)则:$
<!--more-->
$$f(x) = \frac{1}{1+exp(-z)}$$

$\quad \quad$即对于二分类，如果$f(x)\ge{0.5}$,则$x$属于第一类，即预测$y=1$，反之$x$属于第二类，预测$y=0$；样本的分布如下，其中，$C_1$表示第一个类别，$C_2$表示第二个类别，样本个数为$n$


$$trainingData \quad\, x^1 \quad\, x^2 \quad\, x^3 \quad\,\cdots \quad\, x^n $$

$\qquad \qquad \qquad \qquad \qquad \qquad labels \qquad   \quad  C_{1} \quad C_{1} \quad C_{2} \quad \cdots \quad C_{1} \\$
$\qquad$我们的目的是：对于类别为$1$的正样本$f_{w,b}(x)$ 尽可能大,而类别为$2$的负样本$f_{w,b}(x)$ 尽可能小,则我们需要最大化：$L(w,b)=f_{w,b}(x^1)f_{w,b}(x^2)(1-f_{w,b}(x^3))\cdots\ f_{w,b}(x^n)$来寻找最佳的$w$和$b$
$$
w^{\*},b^{\*} = arg\max\limits_{w,b}(L(w,b))\Longrightarrow\ w^{\*},b^{\*} = arg\min\limits_{w,b}(-ln{L(w,b)})
$$

### [随机梯度下降法](https://zh.wikipedia.org/zh-hans/%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95)

$\qquad 我们需要优化的函数:-ln{L(w,b)} = -\{ln{f_{w,b}(x^1)}+lnf_{w,b}(x^2)+ln(1-f_{w,b}(x^3))+\cdots lnf_{w,b}(x^n)\}\quad \\$
$$
\qquad 假设：
\begin{cases}
\hat{y} = 1 \qquad x\in1 \\\
\\
\hat{y} = 0 \qquad  x\in0
\end{cases}
\qquad 已知\,f(x) = \frac{1}{1+exp(-z)}\quad z = \sum_{i=0}^n  w_{i}x_{i} 则
$$
$\qquad 我们需要优化的函数简化为：ln{L(w,b)} =\sum_{j=1}^{n}\{\hat{y}^j\,lnf_{w,b}(x^j)+(1-\hat{y}^j)\,ln(1-f_{w,b}(x^j))\} \\$

$\qquad 当\,\,\hat{y}=1时\quad \hat{y}\,lnf_{w,b}(x)+(1-\hat y)\,ln(1-f_{w,b}(x)) = lnf_{w,b}(x) \\$
$\qquad 当\,\,\hat{y}=0时\quad \hat{y}\,lnf_{w,b}(x)+(1-\hat y)\,ln(1-f_{w,b}(x)) = ln(1-f_{w,b}(x)) \qquad \\$
$\qquad 即均满足上式 , 因此:$

$\qquad \qquad \quad \frac{\partial lnL(w,b)}{\partial w_i}=\sum_{j=1}^{n}\hat{y}^j\frac{ \partial lnf_{w,b}(x^j) }{\partial w_i}+(1-\hat{y}^j)\frac{\partial (1-lnf_{w,b}(x^j))}{\partial w_i} \\$

$\qquad \quad \quad 而 \, \frac{\partial lnf_{w,b}(x)}{\partial w_i}=\frac{\partial lnf_{w,b}(x)}{\partial z}*\frac{\partial z}{\partial w_i} \\$

$\qquad \qquad \qquad \qquad \quad=\frac{1}{f_{w,b}(x)}\* \frac{\partial f_{w,b}(x)}{\partial z}\*x_i \\$

$\qquad \qquad \qquad \qquad \quad=\frac{1}{f_{w,b}(x)}\*f_{w,b}(x)\*(1-f_{w,b}(x))\*x_i \\$

$\qquad \qquad \qquad \qquad \quad=(1-f_{w,b}(x))*x_i \\$

$\quad \quad 同理 \quad   \frac{\partial (1-lnf_{w,b}(x))}{\partial w_i}=f_{w,b}(x)*x_i \qquad 则化简后:\\$
$\qquad \quad\,\, \qquad \frac{\partial lnL(w,b)}{\partial w_i}=\sum_{j=1}^{n}\hat{y}^j\frac{ \partial lnf_{w,b}(x^j) }{\partial w_i}+(1-\hat{y}^j)\frac{\partial (1-lnf_{w,b}(x^j))}{\partial w_i} \\$

$\qquad \qquad \qquad \quad \qquad = \sum_{j=1}^{n}\{\hat{y}^j(1-f_{w,b}(x^j))x^j_i+(1-\hat{y}^j)*f_{w,b}(x^j)x^j_i\} \\$

$\qquad \qquad \quad\qquad \qquad = \sum_{j=1}^{n}(\hat{y}^j -f_{w,b}(x^j))x^j_i \\$

$\qquad b的推导与w的相似，可以得到w的更新迭代过程：w_{i} \leftarrow w_{i}-\alpha*\sum_{j=0}^{n}(\hat{y}^j-f_{w,b}(x^j))x^j_i \\$

![](http://images2017.cnblogs.com/blog/888534/201709/888534-20170908103015851-1635753052.png)

### 思考题
#### 1. 为什么选用$crossEntropy$损失函数，而不用L2损失函数

$答:logistic不像linear \,\, regression使用L2损失函数的原因，主要是由于logistic的funcion的形式，\\$
$由于sigmoid函数的存在，如果logistic采取L2 loss时，损失函数为：\\$
$$\frac{\partial (f_{w,b}(x)-\hat{y})^2}{\partial w_i}=2(f_{w,b}(x)-\hat{y})f_{w,b}(x)(1-f_{w,b}(x))x_i $$
$则当\,\hat{y}=1, f_{w,b}(x) = 1 \quad 预测为1 ，即预测完全正确时 \quad loss=0 \quad  \\$
$但是当\,\hat{y}=1,f_{w,b}(x) = 0 \quad 预测为0 ，即预测完全错误时 \quad loss却依然为0 \quad显然不对 \\$
#### 2. [$logistic \,\,regression$的分类概率为什么选取了$sigmoid$函数](https://www.zhihu.com/question/54707359)
$答: 我们假设样本的分布服从二次高斯分布，即\\$

$f_{\mu,\Sigma}(x) = \frac{1}{(2\pi)^{D/2}}\frac{1}{|\Sigma|^{1/2}}exp\{-\frac{1}{2}(x-\mu)^T|\Sigma|^{-1}(x-\mu)\},其中\mu为均值，\Sigma为协方差矩阵 \\$

$输入为x，输出f_{\mu,\Sigma}(x)为样本x的概率密度，高斯分布的形状分布取决于均值\mu和协方差矩阵\Sigma, \\$
$因此需要求取最佳的高斯分布来满足样本的分布 \\$

$$Maximum Likelihood : L(\mu,\Sigma) = f_{\mu,\Sigma}(x^1)f_{\mu,\Sigma}(x^2)f_{\mu,\Sigma}(x^3)\cdots\cdots\ f_{\mu,\Sigma}(x^{N})$$
$$\mu^{\*}，\Sigma^{\*} = arg\max\limits_{\mu,\Sigma}L(\mu,\Sigma)$$
$$\mu^{\*} = \frac{1}{N}\sum_{i=0}^{N}{x^i}$$
$$\Sigma^{\*} = \frac{1}{N}\sum_{i=0}^{N}{(x^i-\mu^{\*})(x^i-\mu^{\*})^T}$$

$对于一个二分类，我们假设类别1的样本高斯分布的均值为\mu^1,类别2的样本的高斯分布均值为\mu^2,他们具有相同的协方差\Sigma \\$
$$\mu^1 = \sum_{i=1}^{n_1} x_i\qquad (x_i \in C_1) \quad ;\quad \mu^2 = \sum_{i=1}^{n_2} x_i\quad(x_i \in C_2) $$
$$\Sigma^1 = \sum_{i=1}^{n_1}(x_i-u^1)(x_i-u^1)^T ;\quad \Sigma^2 = \sum_{i=1}^{n_2}(x_i-u^2)(x_i-u^2)^T ;\quad \Sigma=\frac{n_1}{n_1+n_2}\Sigma^1+\frac{n_1}{n_1+n_2}\Sigma^2 $$

$对于样本x，如果属于C_1则有：\\$

$\qquad \qquad\qquad \qquad P(C_{1}|x) \,\,= \frac{P(C_{1},x)}{P(x)} \\$

$\qquad \qquad\qquad \qquad \qquad \qquad =\frac{P(x|C_{1})\*P(C_{1})}{P(x|C_{1})\*P(C_{1})+P(x|C_{2})\*P(C_{2})} \\$

$\qquad \qquad\qquad \qquad \qquad \qquad =\frac{1}{1+\frac{P(x|C_{2})P(C_{2})}{P(x|C_{1})P(C_{1})}} \\$

$\qquad \qquad\qquad \qquad \qquad \qquad =\frac{1}{1+exp(-\alpha)} \\$

$其中\,\, \alpha= \ln(\frac{P(x|C_{1})\*P(C_{1})}{P(x|C_{2})\*P(C_{2})})$

$将P(x|C_i)带入高斯分布的公式:\\$
$$P(C_1)=\frac{n_1}{n_1+n_2}\quad , \quad P(C_2)=\frac{n_2}{n_1+n_2} $$
$$P(x|C_1) = \frac{1}{(2\pi)^{D/2}}\frac{1}{|\Sigma|^{1/2}}exp\{-\frac{1}{2}(x-\mu^1)^T|\Sigma|^{-1}(x-\mu^1)\} $$
$$P(x|C_2) = \frac{1}{(2\pi)^{D/2}}\frac{1}{|\Sigma|^{1/2}}exp\{-\frac{1}{2}(x-\mu^2)^T|\Sigma|^{-1}(x-\mu^2)\} $$
$\alpha= lnP(x|C_1)-lnP(x|C_2)+ln\frac{P(C_1)}{P(C_2)} \\$
$\quad =-\frac{1}{2}(x-\mu^1)^T|\Sigma|^{-1}(x-\mu^1)-(-\frac{1}{2}(x-\mu^2)^T|\Sigma|^{-1}(x-\mu^2))+ln\frac{n_1}{n_2}\\$
$\quad =-\frac{1}{2}x^T(\Sigma)^{-1}x+(u^1)^T(\Sigma)^{-1}x-\frac{1}{2}(u^1)^T(\Sigma)^{-1}u^1+\frac{1}{2}x^T(\Sigma)^{-1}x-(u^2)^T(\Sigma)^{-1}x+\frac{1}{2}(u^2)^T(\Sigma)^{-1}u^2+ln\frac{n_1}{n_2}\\$
$\quad = (u^1-u^2)^T(\Sigma)^{-1}x-\frac{1}{2}(u^1)^T(\Sigma)^{-1}u^1+\frac{1}{2}(u^2)^T(\Sigma)^{-1}u^2+ln\frac{n_1}{n_2}\\$
$\quad = wx+b\\$
$\quad w = (u^1-u^2)^T(\Sigma)^{-1} \quad ; \quad b=-\frac{1}{2}(u^1)^T(\Sigma)^{-1}u^1+\frac{1}{2}(u^2)^T(\Sigma)^{-1}u^2+ln\frac{n_1}{n_2}\\$
$\quad 因此可以得到对于满足猜想的二次高斯分布的datasets，生成模型的分类表达式与logistic是一致的 \\$



### 生成模型与判别模型
#### 生成模型
    基于现有的样本，对样本分布做了一个猜测（极大似然），因此当数据集较少，或者有噪声的时候，
都能达到一个较好的结果(不过分依赖于实际样本),并且可以根据不同的概率model完成样本分布的gauss
#### 判别模型
    基于决策的方式（判别式），通过优化方法(sgd)寻找最优参数，对样本的依赖大，样本充足时，其
效果一般比生成模型好(基于事实 not 基于猜测)

### 小扩展
#### 多分类
    基于先验概率得出的每个类别的后验概率为softmax函数，即：
$\\$
$\qquad \qquad \qquad \qquad \, P(C_i|x) = \frac{P(x|C_i)P(C_i)}{\sum_{j=1}^{n}P(x|C_j)P(C_j)}\\$


$\qquad \qquad \qquad \qquad \qquad \qquad = \frac{exp(a_k)}{\sum_{j=1}^{n}a_j}\\$

#### 待续
未完待续

>本文作者： 张峰
>本文链接： https://zhanglaplace.github.io/2017/09/07/logistic/  
>版权声明： 本博客所有文章，均采用 CC BY-NC-SA 3.0 许可协议。转载请注明出处！
