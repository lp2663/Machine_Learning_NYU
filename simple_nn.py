#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 12:32:00 2022

@author: levpaciorkowski
"""

import numpy as np
import matplotlib.pyplot as plt

def ols_nn(X, y, epochs, nu, w_true = None, b_true = None):
    b = 0 if b_true is None else b_true
    w = np.zeros(X.shape[1]) if w_true is None else w_true
    obj_func_values = []
    for _ in range(epochs):
        for i in range(X.shape[0]):
            x = X[i,:]
            pred = np.dot(w, x) + b
            obj_func_values.append((pred-y[i])**2)
            b -= 2*(pred-y[i])*nu
            w -= 2*(pred-y[i])*x*nu
    return b, w, obj_func_values



def f(x, w, b):
    return np.dot(w, x) + b


def simulate_data(dim, n):
    w_true = np.random.random(dim)
    b_true = np.random.randint(-100, 100)
    X = np.random.random(dim)
    y = [f(X, w_true, b_true)]
    for _ in range(n):
        x = np.random.random(dim) * np.random.randint(-100, 100)
        X = np.vstack((X, x))
        y.append(f(x, w_true, b_true))
    return X, y, w_true, b_true
        
    


X_train, y_train, w_true, b_true = simulate_data(15, 100)

b, w, obj_func_values = ols_nn(X_train, y_train, 100, 0.00001)

horizontal = list(range(len(obj_func_values)))


plt.plot(horizontal, obj_func_values)