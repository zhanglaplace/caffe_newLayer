---
title: Linear Regression
date: 2017-09-10 10:23:08
tags: 机器学习
categories: 机器学习
---

## Model and Cost Function(模型和损失函数)
对于model，给出如下定义 $y = \theta x$
损失函数$J(\theta ): minimize\frac{1}{2m}\sum_{i=1}^{m}(h_{\theta}(x^i)-y^i)^2$
<!--more-->
Gradient descent algorithm
repeat until convergence{
    $\quad \theta_j := \theta_j - \alpha\frac{\partial}{\partial \theta_j}J(\theta)$
}


## SVM
寻找两类样本正中间的划分超平面，因为该超平面对训练样本的布局扰动的容忍度最好，是最鲁棒的
划分超平面方程:
$$wx+b = 0$$
我们假使
$$
\begin{cases}
wx_i+b >= 1 \qquad\quad y_i = +1 \\\
\\
wx_i+b <=-1 \qquad\, y_i = -1
\end{cases}
$$
则距离超平面最近的几个点使得下列式子成立
$$\max\limits_{w,b}(\frac{2}{||w||}) \rightarrow \min_{w,b}\frac{1}{2}||w||^2$$
$$s.t. y_i(wx_i+b)\ge 1 i = 1,2,...,m.$$
通用表达式:
    $f(x)=w\psi(x)+b = \sum_{i=1}^{m}a_iy_i\psi(x_i)^T\psi(x)+b=\sum_{i=1}^{m}a_iy_i\kappa(x,x_i)+b$
$\kappa 为核函数.$

>本文作者： 张峰
>本文链接： https://zhanglaplace.github.io/2017/09/10/Linear-Regression/
>版权声明： 本博客所有文章，均采用 CC BY-NC-SA 3.0 许可协议。转载请注明出处！
