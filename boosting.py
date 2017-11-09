import numpy as np
import random
import copy

import math

class BoostingClassifier(object):

    class hn(object):

        def __init__(self,data,y):
            self.idx,self.value_dict = self.get_h(data,y)
            self.data = data
            self.y = y

        def get_h(self,data,y):
            idx = 0
            acc_idx = 0
            value_dict_idx = {}
            for col in range(data.shape[1]):
                acc = 0
                value  = set(data[:,col].tolist())
                value_dict = {}
                for v in value:
                    value_dict[v] =[0,0]
                if len(value)==1:
                    continue
                for line in range(data.shape[0]):
                    value_dict[data[line,col]][0 if y[line]==1 else 1] +=1

                for v in value:
                    if value_dict[v][0] > value_dict[v][1]:
                        acc += value_dict[v][0]
                    else:
                        acc += value_dict[v][1]

                if acc>acc_idx:
                    idx = col
                    value_dict_idx = value_dict

            return idx,value_dict_idx

        def pre(self,d):
            if self.value_dict[d[self.idx]][0] > self.value_dict[d[self.idx]][1] :
                return self.value_dict[d[self.idx]][0]/float(self.value_dict[d[self.idx]][0] +self.value_dict[d[self.idx]][1] )
            else:
                return -1 * self.value_dict[d[self.idx]][1]/float(self.value_dict[d[self.idx]][0] +self.value_dict[d[self.idx]][1] )

        def get_new_D(self):
            # self.data[:,self.idx] = [0] * self.data.shape[0]
            tmp_data =[]
            tmp_y = []
            for id, d in enumerate(self.data.tolist()):
                if self.value_dict[d[self.idx]][0] > self.value_dict[d[self.idx]][1]:
                    if self.y[id] == -1:
                        tmp_data.append(d)
                        tmp_y.append(self.y[id])
                else:
                    if self.y[id] == 1:
                        tmp_data.append(d)
                        tmp_y.append(self.y[id])
            tmp_data = np.array(tmp_data)
            if len(tmp_data)>0:
                tmp_data[:,self.idx] = [0] * tmp_data.shape[0]
            return tmp_data,tmp_y

    def __init__(self):
        self.a = []
        self.h = []

    def fit(self,data,y):
        data_ = copy.deepcopy(data)
        y_ = copy.deepcopy(y)

        for count in range(data.shape[1]):
            n = data_.shape[0]
            if n == 0 :
                break
            self.h.append(self.hn(data_,y))
            data_,y_ = self.h[-1].get_new_D()
            self.a.append(data_.shape[0] /float(n))
        pass

    def predict(self,data):
        if len(self.h)==0:
            raise 'no sulotion!'
        p = 0

        for idx , h in enumerate(self.h):
            p = self.a[idx ] * (h.pre(data))

        return 1 if p>0 else -1




if __name__ =='__main__':
    x = np.array([
        [1,1,1,1,1,1],
        [2,1,1,1,1,1],
        [1,1,2,1,1,1],
        [3,1,1,1,1,1],
        [1,2,1,1,2,2],
        [2,2,1,2,2,2],
        [3,1,1,3,3,2],
        [1,2,1,2,1,1],
        [3,2,2,2,1,1],
        [2,2,1,1,2,2],
        [3,1,1,3,3,1],
        [2,2,1,1,2,1],
        [2,2,2,2,2,1],
        [1,3,3,1,3,2],
        [3,3,3,3,3,1]
    ])
    y = np.array([
        1,1,1,1,1,-1,-1,-1,-1,-1,-1,1,1,-1,-1])
    print (np.array(y))
    c = BoostingClassifier()

    c.fit(np.array(x),y)
    print( 'train finish')
    x = np.array([
        [1,1,2,2,2,1],
        [2,1,2,1,1,1]
    ])
    y = np.array([-1,1])
    yPre = c.predict(x[0])
    print(yPre)
    yPre = c.predict(x[1])
    print(yPre)
    print(y)
    print ('predict finish')
