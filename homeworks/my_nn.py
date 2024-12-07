"""Здесь собраны написанные мной оптимизаторы, элементы и слои,
чтобы можно было их потом импортировать"""

import numpy as np
import torch


class Element:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Element(data={self.data}, grad={self.grad})"

    def __str__(self):
        return f"{self.data})"

    def __add__(self, other):
        if isinstance(other, Element):
            res = Element(self.data + other.data, (self, other), '+')

            def _backward():
                self.grad += 1 * res.grad
                other.grad += 1 * res.grad

            res._backward = _backward
            return res
        else:
            res = Element(self.data + other, tuple([self]), '+')

            def _backward():
                self.grad += 1 * res.grad

            res._backward = _backward
            return res

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + other * (-1)

    def __rsub__(self, other):
        return other + (-1) * self

    def __neg__(self):
        return 0 - self

    def __mul__(self, other):
        if isinstance(other, Element):
            res = Element(self.data * other.data, (self, other), '*')

            def _backward():
                self.grad += other.data * res.grad
                other.grad += self.data * res.grad

            res._backward = _backward
            return res
        else:
            res = Element(self.data * other, tuple([self]), '*')

            def _backward():
                self.grad += other * res.grad

            res._backward = _backward
            return res

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, Element):
            res = Element(self.data / other.data, (self, other), '/')

            def _backward():
                self.grad += 1 / other.data * res.grad
                other.grad += (-1) * self.data / other.data ** 2 * res.grad

            res._backward = _backward
            return res
        else:
            res = Element(self.data / other, tuple([self]), '/')

            def _backward():
                self.grad += 1 / other * res.grad

            res._backward = _backward
            return res

    def __rtruediv__(self, other):
        res = Element(other / self.data, tuple([self]), '/')

        def _backward():
            self.grad += (-1) * other / self.data ** 2 * res.grad

        res._backward = _backward
        return res

    def __pow__(self, power):
        res = Element(self.data ** power, tuple([self]), '**')

        def _backward():
            self.grad += power * self.data ** (power - 1) * res.grad

        res._backward = _backward
        return res

    def __rpow__(self, other):
        res = Element(other ** self.data, tuple([self]), '**')

        def _backward():
            self.grad += other ** self.data * np.log(other) * res.grad

        res._backward = _backward
        return res

    def relu(self):
        res = Element(max(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += res.grad if res.data > 0 else 0

        res._backward = _backward

        return res

    def log(self):
        res = Element(np.log(self.data), tuple([self]), 'log')

        def _backward():
            self.grad += 1 / self.data * res.grad

        res._backward = _backward
        return res

    def backward(self):
        visited = set()
        topo = []

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()


# Функция которая из массива элементов делает массив чисел
def numerize(arr: np.ndarray[Element]):
    new_arr = np.empty_like(arr)
    if len(arr.shape) == 1:
        for i in range(len(arr)):
            new_arr[i] = arr[i].data
        return new_arr
    else:
        for i in range(len(arr)):
            new_arr[i] = numerize(arr[i])
        return new_arr


# Функция которая из массива чисел делает массив элементов
def elementize(arr: np.ndarray):
    new_arr = np.empty_like(arr, dtype=Element)
    if len(arr.shape) == 1:
        for i in range(len(arr)):
            new_arr[i] = Element(arr[i])
        return new_arr
    else:
        for i in range(len(arr)):
            new_arr[i] = elementize(arr[i])
        return new_arr


class HasParameters:
    def __init__(self):
        self.params = []

    def parameters(self):
        return self.params


class BatchNorm1d(HasParameters):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.dim = dim
        # создаем параметры gamma и eps. Они Elements т.к. обучаются
        self.gamma = np.array([Element(data=1)] * dim)
        self.beta = np.array([Element(data=0)] * dim)
        self.eps = eps

        self.params = [self.gamma, self.beta]

    def __call__(self, x):
        # Вычисляем среднее и дисперсию
        # Применяем numerize потому что хотим интерпретировать их как числа
        mu = numerize(x.mean(axis=0))
        var = numerize(1 / (self.dim - 1) * np.sum((x - mu) ** 2, axis=0))
        # Нормализовываем и применяем преобразование
        x_normalized = (x - mu) / (var + self.eps) ** (1 / 2)
        y = self.gamma * x_normalized + self.beta
        return y


class Linear(HasParameters):
    def __init__(self, in_features, out_features, gain=np.sqrt(2)):
        super().__init__()
        # Параметры - матрица весов W и вектор сдвига b
        # инициализируем случайными весами от -0.5 до 0.5
        std = gain * np.sqrt(2 / (in_features + out_features))
        self.W = elementize(np.random.normal(loc=0, scale=std, size=(in_features, out_features)))
        self.b = elementize(np.zeros(out_features) + 0.01)

        self.params = [*self.W.flatten().tolist(), *self.b.tolist()]

    def __call__(self, x):
        # Спасибо numpy за матричные операции!!
        return x @ self.W + self.b


class Dropout:
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, x):
        # Строки - батчи, столбцы - нейроны. Мы генерируем строку 0 или 1 с вероятностью p
        # умножив x на эту строку мы "отключим" каждый нейрон с вероятностью p
        mask = (np.random.rand(x.shape[1]) > self.p).astype(int)
        return x * mask


class ReLU:
    def __call__(self, x):
        # создаем выход того же размера
        res = np.empty_like(x, dtype=Element)
        if len(x.shape) == 1:
            for i in range(len(x)):
                # дергаем relu у элемента
                res[i] = x[i].relu()
            return res
        else:
            for i in range(len(x)):
                res[i] = self(x[i])
            return res


def tanh(x):
    return (np.e ** (2 * x) - 1) / (np.e ** (2 * x) + 1)


class Tanh:
    def __call__(self, x):
        res = np.empty_like(x, dtype=Element)
        if len(x.shape) == 1:
            for i in range(len(x)):
                res[i] = tanh(x[i])
            return res
        else:
            for i in range(len(x)):
                res[i] = self(x[i])
            return res


def sigmoid(x):
    return 1 / (1 + np.e ** (-x))


class Sigmoid:
    def __call__(self, x):
        y = np.empty_like(x)
        if len(x.shape) == 1:
            for i in range(len(x)):
                y[i] = sigmoid(x[i])
            return y
        else:
            for i in range(len(x)):
                y[i] = self(x[i])
            return y


def softmax(x):
    return np.e ** x / np.sum(np.e ** x)


class Softmax:
    def __call__(self, x):
        # построчно вызываем softmax
        res = np.empty_like(x)
        for i in range(len(x)):
            res[i] = softmax(x[i])
        return res


class Module(HasParameters):
    def __setattr__(self, key, value):
        if isinstance(value, HasParameters):
            self.params += value.params
        elif hasattr(value, '__iter__'):
            for param in value:
                if isinstance(param, HasParameters):
                    self.params += param.params
        object.__setattr__(self, key, value)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args):
        return self.forward(*args)


class Adam:
    # теперь принимаем параметры в момент инициализации, а не при вызове step
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        # делаем из генератора список для простоты и вызываем retain_grad чтобы все было хорошо
        self.params = list(params)
        self.m = {}
        self.s = {}

    # теперь не принимает на вход параметры и градиенты
    def step(self):
        if not self.m:
            self.m = [np.zeros_like(p) for p in self.params]
            self.s = [np.zeros_like(p) for p in self.params]
        # добавляем no_grad
        with torch.no_grad():
            for i, param in enumerate(self.params):
                # получаем параметр и высчитанный градиент
                p = param
                g = param.grad
                m = self.m[i]
                s = self.s[i]

                m = self.beta1 * m + (1 - self.beta1) * g
                s = self.beta2 * s + (1 - self.beta2) * (g ** 2)

                p -= self.lr * m / (np.sqrt(s) + self.eps)

                self.m[i] = m
                self.s[i] = s

    # написали свой zero_grad
    def zero_grad(self):
        for param in self.params:
            param.grad = 0


class Dataloader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = len(self.dataset) // self.batch_size + (0 if len(self.dataset) % self.batch_size == 0 else 1)
        self.permute = np.random.permutation(range(len(self.dataset))) if shuffle else range(len(self.dataset))

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for i in range(len(self.dataset) // self.batch_size):
            yield np.vstack([self.dataset[self.permute[j + i * self.batch_size]] for j in range(self.batch_size)])
        delta = len(self.dataset) % self.batch_size
        if delta != 0:
            yield np.vstack([self.dataset[self.permute[j + len(self.dataset) // self.batch_size * self.batch_size]] for j in range(delta)])
