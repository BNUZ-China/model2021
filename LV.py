import numpy as np
from scipy.special import comb

#——————————————Gillespie———————————————
class Reaction: # 封装的类，代表每一个化学反应
    def __init__(self, rate=0., num_lefts=None, num_rights=None):
        self.rate = rate # 反应速率
        assert len(num_lefts) == len(num_rights)
        self.num_lefts = np.array(num_lefts) # 反应前各个反应物的数目
        self.num_rights = np.array(num_rights) # 反应后各个反应物的数目
        self.num_diff = self.num_rights - self.num_lefts # 改变数
    def combine(self, n, s): # 算组合数
        return np.prod(comb(n, s))
    def propensity(self, n): # 算反应倾向函数
        return self.rate * self.combine(n, self.num_lefts)

class System: # 封装的类，代表多个化学反应构成的系统
    def __init__(self, num_elements):
        assert num_elements > 0
        self.num_elements = num_elements # 系统内的反应物的类别数
        self.reactions = [] # 反应集合
    def add_reaction(self, rate=0., num_lefts=None, num_rights=None):
        assert len(num_lefts) == self.num_elements
        assert len(num_rights) == self.num_elements
        self.reactions.append(Reaction(rate, num_lefts, num_rights))
    def evolute(self, steps, inits=None): # 模拟演化
        self.t = [0] # 时间轨迹，t[0]是初始时间
        if inits is None:
            self.n = [np.ones(self.num_elements)]
        else:
            self.n = [np.array(inits)] # 反应物数目，n[0]是初始数目
        for i in range(steps):
            A = np.array([rec.propensity(self.n[-1])
                          for rec in self.reactions]) # 算每个反应的倾向函数
            A0 = A.sum()
            A /= A0 # 归一化得到概率分布
            t0 = -np.log(np.random.random())/A0 # 按概率选择下一个反应发生的间隔
            self.t.append(self.t[-1] + t0)
            d = np.random.choice(self.reactions, p=A) # 按概率选择其中一个反应发生
            self.n.append(self.n[-1] + d.num_diff)


#————————————Naive Bayes—————————————————
def prior(vd,normal):
    for i in range(len(normal)):
        for j in range(vd):
            normal_count_dict[j]=normal_count_dict.get(j,0)+normal.content[i][j] 
            #计算vd各特征向量出现的次数，即，求content各特征向量count按列累加的结果
 
    for j in range(vd):
        normal_count_prob[j]=(normal_count_dict.get(j,0)+1)/(len(normal)+2)
        #计算vd各特征向量出现的概率（拉普拉斯平滑）
  
#定义后验概率
def posterior():
    for i in range(X_test_array.shape[0]):
        for j in range(X_test_array.shape[1]):
            if X_test_array[i][j]!=0:
                Px_spam[i]=Px_spam.get(i,1)*spam_count_prob.get(j)
                #计算vd条件下，各测试样本的后验概率
                Px_normal[i]=Px_normal.get(i,1)*normal_count_prob.get(j)
                #计算normal条件下，各测试样本的后验概率
 
        test_classify[i]=Py0*Px_normal.get(i,0),Py1*Px_spam.get(i,0)
                 #后验概率P(Y|X)=P(Y)*P(X|Y)/P(X)


#——————————————bacteria——————————————————
class bacteria:
    def __init__(self, self_name):
        self.bac_name = self_name
    
#————————————————main———————————————————

