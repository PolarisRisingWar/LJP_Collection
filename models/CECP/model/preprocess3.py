#预处理代码

import pickle as pk
import numpy as np

path='/data/wanghuijuan/cecp_data/'

path_elements = path + 'elements_criminal.pkl'
with open(path_elements, 'rb') as f:
    data = pk.load(f)

print(data.keys())
print(data['ele_subject'].shape)  #[罪名数,100]
print(data['ele_subjective'].shape)
print(data['ele_object'].shape)
print(data['ele_objective'].shape)
print(data['num2charge'].keys())  #这个捏就是ID与罪名文本的对应啦，啊我的问题就是这玩意对应起来有意义吗
print(data['num2charge'][0])

for k in ['train','valid','test']:
    path='/data/wanghuijuan/cecp_data/cail_small_'+k+'.pkl'
    with open(path, 'rb') as f:
        data = pk.load(f)
    print(np.unique(data['y']))