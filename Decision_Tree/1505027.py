import pandas as pd
import numpy as np
from pprint import pprint
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import KBinsDiscretizer  
import math
import imblearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek


## TELCO DATASET

def loadTelcoData(label):
    dataframe = pd.read_csv('Data/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv', delimiter=",")
    # non_categorical =['tenure','MonthlyCharges','TotalCharges']
    
    dataframe = dataframe.drop('customerID', axis=1)
    dataframe = dataframe.dropna(axis=0, subset=[label])
    dataframe = dataframe.reset_index(drop=True)
    
    print ("\nMissing values :  ", dataframe.isnull().sum().values.sum())
    
    # null_columns=dataframe.columns[dataframe.isnull().any()]
    dataframe = dataframe.applymap(lambda x: np.nan if isinstance(x, str) and x.isspace() else x)
    dataframe["TotalCharges"] = dataframe["TotalCharges"].astype(float)

    numeric_cols = dataframe._get_numeric_data().columns
    print(numeric_cols)

    print("Before missing values: ", dataframe['TotalCharges'].isnull().sum().sum())
        

    median = dataframe['TotalCharges'].median()
    dataframe['TotalCharges'].fillna(median, inplace=True)
    
    print("After missing values: ", dataframe['TotalCharges'].isnull().sum().sum())
    
    le = LabelEncoder()

    dataframe[label] = le.fit_transform(dataframe[label])
    
    ohe = ce.OneHotEncoder(handle_unknown='ignore', use_cat_names=True)
    dfOhe = ohe.fit_transform(dataframe)
    
    for col in numeric_cols:
        if(col == 'SeniorCitizen'):
            continue
        est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
        dfOhe[col] = est.fit_transform(dfOhe[[col]]) 
    
    return dfOhe



## ADULT DATASET


def loadAdultData(label):
    dataframe = pd.read_csv('Data/Adult/adult-income-dataset/adult.csv', delimiter=",")

    dataframe[label] = dataframe[label].replace('?', pd.np.nan)    
    dataframe = dataframe.dropna(axis=0, subset=[label])
    dataframe = dataframe.reset_index(drop=True)
    
    le = LabelEncoder()

    dataframe[label] = le.fit_transform(dataframe[label])
    
    
    numeric_cols = dataframe._get_numeric_data().columns
    print(numeric_cols)
    cat_cols = [x for x in dataframe.columns if x not in numeric_cols]

    
    for col in cat_cols:
        uVal, occurances = np.unique(dataframe[col], return_counts = True)
        most_freq_attrib = uVal[np.argmax(occurances, axis = 0)]
        dataframe[col][dataframe[col] == '?'] = most_freq_attrib

    dataframe = dataframe.replace('?', pd.np.nan)
        
    for col in numeric_cols:
        median = dataframe[col].median()
        dataframe[col].fillna(median, inplace=True)

    ohe = ce.OneHotEncoder(handle_unknown='ignore', use_cat_names=True)
    dfOhe = ohe.fit_transform(dataframe)
    
    for col in numeric_cols:
        if(col == label):
            continue
        est = KBinsDiscretizer(n_bins=13, encode='ordinal', strategy='uniform')
        dfOhe[col] = est.fit_transform(dfOhe[[col]]) 
    return dfOhe



## CREDIT CARD FRAUD DATASET

def loadCreditCardData(label):
    dataframe = pd.read_csv("Data/creditcardfraud/creditcard.csv", delimiter=",",engine='python')
    
    dataframe = dataframe.dropna(axis=0, subset=[label])
    dataframe = dataframe.reset_index(drop=True)
    
    quantile_scaler = QuantileTransformer(random_state=0, output_distribution='uniform')
    
    Scaled_amount = quantile_scaler.fit_transform(dataframe['Amount'].values.reshape(-1, 1))
    Scaled_time = quantile_scaler.fit_transform(dataframe['Time'].values.reshape(-1, 1))
    
    dataframe.drop(['Time','Amount'], axis=1, inplace=True)

    dataframe.insert(0, 'Amount', Scaled_amount)
    dataframe.insert(1, 'Time', Scaled_time)
    
    # target_col = label
    # other_cols = [x for x in dataframe.columns if x not in target_col]
        
    Y_true = dataframe.loc[dataframe[label] == 1]
    Y_false = dataframe.loc[dataframe[label] == 0]
    Y_false = Y_false.sample(n=20000)
    subdata = Y_true.append(Y_false, ignore_index=True)
    dataframe = subdata.sample(frac=1)
    
    
    dataframe = dataframe.reset_index(drop=True)
    
    print('No Frauds', round(dataframe[label].value_counts()[0]), ' of the dataset')
    print('Frauds', round(dataframe[label].value_counts()[1]), ' of the dataset')
    
    numeric_cols = dataframe._get_numeric_data().columns
    # cat_cols = [x for x in dataframe.columns if x not in numeric_cols]

    for col in numeric_cols:
        median = dataframe[col].median()
        dataframe[col].fillna(median, inplace=True)
        
    
    for col in numeric_cols:
        if(col == label):
            continue
        est = KBinsDiscretizer(n_bins=13, encode='ordinal', strategy='uniform')
        dataframe[col] = est.fit_transform(dataframe[[col]]) 
    
    return dataframe


## MODEL


def convertDecisionRange(data):
    for i in range(len(data)):
        if data[i] == 0:
            data[i] = -1
    return data

def Entropy(attr):
    uVal, occurances = np.unique(attr, return_counts = True)
    esum = 0
    for v in range(len(uVal)):
        ext = (-1)*(occurances[v]/np.sum(occurances))*np.log2(occurances[v]/np.sum(occurances))
        esum = esum + ext
    return esum


def IG(data, split_attr, result_attr):
    parEntropy = Entropy(data[result_attr])
    uVal, occurances = np.unique(data[split_attr], return_counts = True)
    esum = 0
    for v in range(len(uVal)):
        tempdata = data.where(data[split_attr]==uVal[v]).dropna()
        esum = esum + ((occurances[v]/np.sum(occurances))*Entropy(tempdata[result_attr]))
    IG = parEntropy - esum
    return IG

def Plurality_value(parent_data, result_attr):
    idx = np.argmax(np.unique(parent_data[result_attr], return_counts=True)[1])
    return np.unique(parent_data[result_attr])[idx]


def Decision_Tree_Learning(data, attributes, result_attr, parent_data, depth, max_depth):
    if(len(data)==0):
        return Plurality_value(parent_data, result_attr)
    elif(len(np.unique(data[result_attr])) <= 1):
        return np.unique(data[result_attr])[0]
    elif((len(attributes)==0) or (depth==max_depth)):
        return Plurality_value(data, result_attr)
    else:
        igvals = []
        track = 0
        for attribute in attributes:
            tval = IG(data,attribute,result_attr)
            track += 1
            igvals.append(tval)
            
        
        Max_IG = attributes[np.argmax(igvals)]
        tree = {Max_IG:{}}
        
        temp = []        
        for i in attributes:
            if(i != Max_IG):
                temp.append(i)
        attributes = temp

        unique_features = np.unique(data[Max_IG])
        
        for feature in unique_features:
            split_data = data.where(data[Max_IG] == feature)
            split_data.dropna(inplace=True)
            subtree = Decision_Tree_Learning(split_data,attributes,result_attr,data,(depth+1),max_depth)
            tree[Max_IG][feature] = subtree
            
        return tree  



def get_prediction(sample, dt, leaf = 1):
    sample_nodes = list(sample.keys())
    tree_nodes = list(dt.keys())
    
    for node in sample_nodes:
        if(node in tree_nodes):
            try:
                check = dt[node][sample[node]] 
            except:
                return leaf
            if(isinstance(dt[node][sample[node]],dict)):
                return get_prediction(sample,dt[node][sample[node]])
            else:
                return dt[node][sample[node]] 



def accuracy_test(data,tree, label):
    records = data.iloc[:,:-1].to_dict(orient = "records")
    pred = []
    for i in range(len(data)):
        val = int(get_prediction(records[i],tree,1.0))
        pred.append(val)
    
    pred = [int(i) for i in pred]
    
    true = data[label].values
    count = 0
    for i in range(len(data)):
        if(pred[i] == int(true[i])):
            count += 1
    accuracy = (count*100.0)/(len(data))
    print('The prediction accuracy is:',accuracy,'%')

    return pred, data


def adaboostPredict(dataframe,h,z,label):
    true_label = dataframe[label].values
    N = dataframe.shape[0]
    
    true_label = convertDecisionRange(true_label)

    size = len(h)
    vect = [0 for i in range(N)]
    
    for i in range(size):
        records = dataframe.iloc[:,:-1].to_dict(orient = "records")
        for j in range(N):
            if(get_prediction(records[j],h[i],1.0) == 1):
                vect[j] += z[i]
            else:
                vect[j] -= z[i]
    
    
    final_pred = []
    for r in vect:
        final_pred.append(1 if r > 0.0 else 0)

    final_pred = convertDecisionRange(final_pred)
        
    count = 0
    for i in range(N):
        if(final_pred[i] == int(true_label[i])):
            count += 1
    accuracy = (count*100.0)/(N)
    print('Accuracy:',accuracy,'%')
    
    return final_pred



def Normalize(w):
    if(sum(w) == 0):
        w = [1 for i in w]
    else:
        w = [float(i) / sum(w) for i in w]
    return w


def AdaBoost(data, attributes, result_attr, K):
    N = data.shape[0]
    w = [float(1)/N]*N
    z = []
    h = []
    
    df = data.copy()

    depth = 0
    
    while depth < K:
        dfs = df.sample(n = N, weights=w, replace=True)
        dt = Decision_Tree_Learning(dfs, attributes, result_attr, data, 0, 1)
        error = 0.0
        predicted, tdata = accuracy_test(data.copy(), dt, result_attr)
        tdata = tdata[result_attr].values
        for i in range(N):
            if(predicted[i] != tdata[i]):
                error += w[i]
        if(error >= 0.5):
            continue
        
        for i in range(N):
            if((predicted[i] == tdata[i])):
                w[i] = w[i] * error/(1.0-error)
        
        w = Normalize(w)
        
        h.append(dt)
        if(error == 0):
            z.append(math.log(float("inf"),2))
        else:
            z.append(math.log(((1-error)/error),2))
        
        depth += 1
    
    return h,z



def PerformanceEvaluation(test_Y, pred_Y):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(test_Y)):
        if pred_Y[i] == 1:
            if test_Y[i] == 1:
                TP += 1
            else:
                FP += 1

        else:
            if test_Y[i] == 1:
                FN += 1
            else:
                TN += 1

    accuracy = ((TP+TN)*1.0)/len(test_Y)
    Recall = (TP*1.0)/(TP+FN)
    TNR = (TN*1.0)/(TN+FP)
    Precision = (TP*1.0)/(TP+FP)
    FDR = (FP*1.0)/(FP+TP)
    F1Score = ((Precision * Recall * 1.0)/(Precision + Recall)) * 2.0

    print('True Positive:',TP)
    print('False Positive:',FP)
    print('False Negative:',FN)
    print('True Negative:',TN)
    print('Accuracy:',accuracy*100,'%')
    print('Recall:',Recall)
    print('Ture Negative rate:',TNR)
    print('Precision:',Precision)
    print('False Discovery rate:',FDR)
    print('F1 Score:',F1Score)




## TRAINING


## Dataset splitting 
label = 'Churn'
df_telco = loadTelcoData('Churn')
train_data,test_data = train_test_split(df_telco,test_size = .20 ,random_state = 111)
# train_data,test_data = train_test_split(df_adult, test_size = .20, random_state = 111)
# train_data,test_data = train_test_split(df_credit,test_size = .20 ,random_state = 111)
td_dt = train_data.copy()
ts_dt = test_data.copy()


## Training


## Simple DT
tree = Decision_Tree_Learning(train_data.copy(),train_data.columns[:-1], label, train_data.copy(), 0, 5)
print("Tree built")
predicted, data = accuracy_test(train_data.copy(),tree,label)

## AdaBoost 
h, z = AdaBoost(train_data.copy(),train_data.columns[:-1], label, 5)
pred_test = adaboostPredict(train_data.copy(),h,z,label)



## Performance Evaluation

test_Y = data[label].values
pred_Y = predicted
#pred_Y = pred_Y.astype(int)


true_Y = convertDecisionRange(test_Y)
pred_Y = convertDecisionRange(pred_Y)

#Decision Tree
PerformanceEvaluation(true_Y, pred_Y)

#Adaboost
PerformanceEvaluation(true_Y, pred_test)
