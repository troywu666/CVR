#!/usr/bin/env python
# coding: utf-8

# # 数据合并处理

# In[ ]:


import pandas as pd
import numpy as np
import scipy as sp

def read_csv_file(f, logging = False):
    data = pd.read_csv(f)
    if logging:
        print(data.head())
        print(data.columns.values)
        print(data.describe())
        print(data.info())
    return data

def categories_process_first_class(cate):
    cate = str(cate)
    return int(cate[0])
        
    
def categories_process_second_class(cate):
    cate = str(cate)
    if len(cate) < 3:
        return 0
    else:
        return int(cate[1:])
    
def age_process(age):
    age = int(age)
    if age == 0:
        return 0
    if age < 16:
        return 1
    if age < 23:
        return 2
    if age < 27:
        return 3
    else:
        return 4

def province_process(hometown):
    hometown = str(hometown)
    if len(hometown) == 1:
        return 0
    else:
        return int(hometown[: -2])

def city_process(hometown):
    hometown = str(hometown)
    if len(hometown) == 1:
        return 0
    else:
        return int(hometown[-2: ])

def get_time_day(t):
    t = str(t)
    t = int(t[0: 2])
    return t

def get_time_hour(t):
    t = str(t)
    t = int(t[2: 4])
    if t < 6:
        return 0
    if t < 12:
        return 1
    if t < 18:
        return 2
    else:
        return 3

def get_time_min(t):
    t = str(t)
    t = int(t[4: ])
    return t

def logloss(act, pred):
    epsilon = 1e-5
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - eplison, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0/len(act)
    return ll


# In[ ]:


from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

print('test dataset')
test_data = read_csv_file('./pre/test.csv', logging = True)
print('================================================\n')
print('train dataset:')
train_data = read_csv_file('./pre/train.csv', logging = True)
print('================================================\n')
print('ad dataset')
ad = read_csv_file('./pre/ad.csv', logging = True)
print('================================================\n')
print('app dataset')
app_categories = read_csv_file('./pre/app_categories.csv', logging = True)
print('================================================\n')
print('user dataset')
user = read_csv_file('./pre/user.csv', logging = True)
print('================================================\n')
print('position dataset')
position = read_csv_file('./pre/position.csv', logging= True)
print('================================================\n')
print('user_app_actions dataset')
user_app_actions = read_csv_file('./pre/user_app_actions.csv', logging = True)


# In[ ]:


print('================================================\n')
print('user_installedapps dataset')
user_installedapps = read_csv_file('./pre/user_installedapps.csv', logging = False)


# In[ ]:


train = train_data.drop(['label', 'conversionTime'], axis = 1)
test = test_data.drop(['label', 'instanceID'], axis = 1)
data = pd.concat((train, test), axis = 0)


# In[ ]:


print(data.shape)
print(train.shape)
print(test.shape)


# In[ ]:


data.isnull().sum()


# In[ ]:


data_ad = data.merge(ad, on = 'creativeID', how = 'left')
data_ad.shape


# In[ ]:


data_ad['click_day'] = data_ad['clickTime'].apply(get_time_day)
data_ad['click_min'] = data_ad['clickTime'].apply(get_time_min)
data_ad['click_hour'] = data_ad['clickTime'].apply(get_time_hour)
data_ad.drop(['clickTime'], inplace = True, axis = 1)
data_ad.tail()


# In[ ]:


data_ad_app = data_ad.merge(app_categories, on = 'appID', how = 'left')
data_ad_app['app_first_categories'] = data_ad_app['appCategory'].apply(categories_process_first_class)
data_ad_app['app_second_categories'] = data_ad_app['appCategory'].apply(categories_process_second_class)
data_ad_app.drop(['appCategory'], axis = 1, inplace = True)


# In[ ]:


data_ad_app_user = data_ad_app.merge(user, on = 'userID', how = 'left')


# In[ ]:


##test数据集的age数据缺失较多
data_ad_app_user['age'].describe()


# In[ ]:


data_ad_app_user['age'].replace(0.0, data_ad_app_user['age'][data_ad_app_user['age'] != 0.0].mean(), inplace = True)


# In[ ]:


data_ad_app_user['age'].fillna(data_ad_app_user['age'][data_ad_app_user['age'] != 0.0].mean(), inplace = True)
data_ad_app_user['age'].isnull().sum()


# In[ ]:


data_ad_app_user['age'] = data_ad_app_user['age'].apply(age_process)


# In[ ]:


data_ad_app_user['residence_province'] = data_ad_app_user['residence'].apply(province_process)
data_ad_app_user['residence_city'] = data_ad_app_user['residence'].apply(city_process)
data_ad_app_user.drop(['residence'], inplace = True, axis = 1)
data_ad_app_user.tail()


# In[ ]:


data_ad_app_user_position = data_ad_app_user.merge(position, on = 'positionID', how = 'left')


# In[ ]:


x_train = data_ad_app_user_position[: 3749528]
y_train = train_data['label']
test = data_ad_app_user_position[3749528: ]


# In[ ]:


import pickle
with open('values.pkl', 'wb') as f:
    pickle.dump(x_train, f)
    pickle.dump(y_train, f)
    pickle.dump(test, f)


# # 数据建模

# ## 导入数据

# In[4]:


import pickle
with open('values.pkl', 'rb') as f:
    x_train = pickle.load(f)
    y_train = pickle.load(f)
    test = pickle.load(f) 


# In[3]:


y_train.value_counts()
##正负样本非常不均衡，但由于过采样后数据量过大，电脑内存有限，使用欠采样的方法


# ## 过采样后存储数据

# In[6]:


from imblearn.over_sampling import ADASYN

x_train_oversam, y_train_oversam = ADASYN().fit_sample(x_train, y_train)


# In[7]:


import pickle
with open('values_resam.pkl', 'wb') as f:
    pickle.dump(x_train_oversam, f)
    pickle.dump(y_train_oversam, f)   


# In[8]:


print(x_train_oversam.shape)
print(y_train_oversam.shape)


# In[ ]:


## EasyEnsemble
'''
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

bbc = BalancedBaggingClassifier(
    base_estimator = DecisionTreeClassifier(),
    ratio = 'auto',
    replacement = False,
    random_state = 100,
    n_jobs = -1
)

kf = KFold(n_splits = 5)
for train_index, val_index in kf.split(x_train, y_train):
    X_train,Y_train = x_train[train_index, : ], y_train[train_index]
    X_val, Y_val = x_train[val_index, : ], y_train[val_index]
    bbc.fit(X_train, Y_train)
    pred_bbc = bbc.predict_proba(X_val)
    print(pred_bbc)
    print('The logloss is ', logloss(Y_val, pred_bbc))
'''


# ## 欠采样并存储数据

# In[ ]:


import numpy as np
import pickle
from imblearn.ensemble import BalanceCascade
from sklearn.svm import SVC

with open('values.pkl', 'rb') as f:
    x_train = pickle.load(f)
    y_train = pickle.load(f)
    test = pickle.load(f)
    
x_train = np.array(x_train)
y_train = np.array(y_train)
test = np.array(test)

bc = BalanceCascade(
    estimator = SVC(gamma = 'auto'),
    random_state = 100,
    n_max_subset = 5
)
x_train_resam, y_train_resam = bc.fit_sample(x_train, y_train)

with open('values_undersampling.pkl', 'wb') as f:
    pickle.dump(x_train_resam, f)
    pickle.dump(y_train_resam, f)
    pickle.dump(test, f)


# In[13]:


with open('values_undersampling.pkl', 'rb') as f:
    x_train_resam = pickle.load(f)
    y_train_resam = pickle.load(f)
    test = pickle(f)


# ### Xgbosst

# In[ ]:


import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import numpy as np

param_grid = {
    'max_depth': [3, 4, 5, 8, 10],
    'n_estimators': [50, 100, 200, 400, 600, 800, 1000],
    'laerning_rate': [0.1, 0.2, 0.3],
    'gamma': [0, 0.2],
    'subsample': [0.8, 1],
             }

xgb_model = xgb.XGBClassifier()
rgs = GridSearchCV(xgb_model, param_grid, n_jobs = -1)
rgs.fit(x_train_resam, y_trian_resam)
print(rgs.best_score_)
print(rgs.best_params_)
pred = rgs.predict_proba(test)
print('The logloss is ', logloss(Y_val, pred_bbc))


# ### LightGBM

# In[ ]:




