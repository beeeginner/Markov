import numpy as np
from sklearn.model_selection import train_test_split


class MarkovClassifier0:
    def __init__(self,metric='Cosine',step=0,p=0.5):
        '''
        复现了论文Classification with Graph-Based Markov Chain
        y=YP(t)W(+)w(V,v)
        :param metric:计算相似度指标的方式，文本数据使用余弦相似度，数量数据使用欧几里得相似度
        :param step: 随机游走的次数
        :param norm: 如果选用闵可夫斯基距离，指定的范数
        '''
        assert metric in ('Cosine','Euclidean','pnorm'),f'不支持的相似度 {metric},支持的相似度: Euclidean,Cosine,pnorm'
        self.metric=metric
        self.step=step
        self.p=p

        if self.metric=='Cosine':
            self._f = self._Cosinesim
        elif self.metric=='Euclidean':
            self._f = self._Euclideansim
        elif self.metric=='pnorm':
            self._f = self._norm_p
        self._numclasses = None
        self.P = None
        self.W = None
        self.W_plus = None

    # 类别的编码解码器
    def _one_hot_decoding(self, one_hot):
        return np.argmax(one_hot)

    def _one_hot_encoding(self, label):
        if not isinstance(label, np.int32):
            label = np.int32(label)
        res = [0] * self._numclasses
        res[label] = 1
        return np.array(res, dtype=np.float32)

    # 将输入的一系列类别标签转换成矩阵表示
    def _convert_Y(self, y):

        self._numclasses = len(np.unique(y))
        res = np.empty((self._numclasses, y.shape[0]), dtype=np.float32)
        for id, label in enumerate(y):
            res[:, id] = self._one_hot_encoding(label)
        return res

    def _similarity(self,data):
        num_points = data.shape[0]
        distance_matrix = np.zeros((num_points, num_points))
        for i in range(num_points):
            for j in range(num_points):
                distance_matrix[i, j] = 1/(self._f(data[i],data[j])+1) if self.metric=='pnorm' else self._f(data[i],data[j])
        return distance_matrix

    def fit(self, X):
        '''

        :param data: 包括训练数据和测试数据的所有数据
        '''
        W=self._similarity(X)
        self.W_plus = np.linalg.pinv(W)
        if self.step!=0:
            self.P = np.linalg.inv(np.diag(W @ np.ones(W.shape[0]))) @ W
            self.P = self.P / np.sum(self.P, axis=0)
        print('finished trainning!')

    def _Euclideansim(self, a, b):
        return 1 / (1 + np.linalg.norm(a - b))

    def _Cosinesim(self, a, b):
        return a@b/np.linalg.norm(a)/np.linalg.norm(b)

    def _norm_p(self,x1, x2):
        p=self.p
        return np.power(np.sum(np.power(np.abs(x1 - x2),p)),(1 / p))

    def generate_predict(self, X_train, y_train, x_test):

        for x in x_test:
            sim = np.zeros(X_train.shape[0])
            for id in range(X_train.shape[0]):
                sim[id]=1/(self._f(x,X_train[id]) +1) if self.metric=='pnorm' else self._f(x,X_train[id])
            res = self._convert_Y(y_train)
            for i in range(self.step):
                res = res @ self.P
            res = res @ self.W_plus @ sim
            yield self._one_hot_decoding(res)