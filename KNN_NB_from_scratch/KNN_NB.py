#!/usr/bin/env python
# coding: utf-8

# In[31]:


from tqdm import tqdm_notebook as tqdm
tqdm().pandas()


# In[32]:


get_ipython().system('pip install tweet-preprocessor')
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# In[33]:


from IPython.display import Audio
sound_file = 'E:/sound/beep.wav'


# In[34]:


# !pip3 install bs4
from bs4 import BeautifulSoup as bs
import pickle
import numpy as np
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import preprocessor as p
import re
import pandas as pd
from scipy.stats import ttest_rel


# # Text Preprocessing

# In[35]:


uri_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'

def remove_tags(text):
    return ''.join(et.fromstring(text).itertext())

def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def clean_text(text):
    #Remove HTML tags
    text = bs(text.replace("\n", " "), "html.parser")#.text
    
    if text.code:
        text.code.decompose()
    
    #Get all the text out of the html
    text =  text.get_text()
    
    #Remove URLS, mentions, hashtags, Reserved Words, eomji, smiley, numbers
    text = p.clean(text)
    
    #Remove emojis
    text = remove_emoji(text)
    
    #Returning text stripping out all uris
    text = re.sub(uri_re, "", text)
    
    #Remove leading and trailing spaces
    text = text.strip()
    
    #Conversion to lower case
    text = text.lower()
    

    #Punctuation removal
    text = text.translate((str.maketrans('','',string.punctuation)))
    
    #Tokenization
    text = word_tokenize(text) 
    
    #Stopwords removal
    stopwords_verbs = ['save', 'say', 'get', 'go', 'know', 'may', 'need', 'like', 'make', 'see', 'want', 'come', 'take', 'use', 'would', 'can', 'could']
    stopwords_other = ['one', 'ive', 'ill', 'im', 'etc', 'rt', 'thing', 'time', 'de', 'en', 'us', 'also', 'something']
    my_stopwords = stopwords.words('english') + stopwords_verbs + stopwords_other
    text = [word for word in text if not word in my_stopwords]
    
    #Lemmatization
    lemmatizer=WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]

    #Stemming
    stemmer= PorterStemmer()
    text = [stemmer.stem(word) for word in text]
    
    #Remove small words and digits
    text = [word for word in text if len(word)>1]
    text = [x for x in text if not (x.isdigit() 
                                         or x[0] == '-' and x[1:].isdigit())]
    
    return text


# In[36]:


import pandas as pd 
import xml.etree.ElementTree as et 
import numpy as np
import os

def get_df(filename, name, target):
    out_df = pd.DataFrame()
    filetype = name[:-4]
    with open(filename,'r',encoding='utf-8') as xml_file:
        xtree = et.parse(xml_file)
        xroot = xtree.getroot() 
        df_cols = ["Body", "Type", "Label"]
        rows = []
        for node in xroot: 
            body = node.attrib.get("Body") if node is not None else np.NaN
            rows.append({"Body": body, "Type": filetype, "Label": target})

        out_df = pd.DataFrame(rows, columns = df_cols)
        out_df.dropna(subset=['Body'], inplace=True)
        idx = 0
        sents = []
        for sentences in tqdm(out_df['Body']):
          if(idx == 1200):
            break
          cleaned_text = clean_text(sentences)
          if(len(cleaned_text)==0):
            continue
          sents.append(cleaned_text)
          idx += 1
        
        out_df = out_df[:1200]
        out_df['sentences'] = sents
    return out_df


def combine_dataset(directory, type_name, topics):
    itr = 0
    files = ["Coffee","Arduino","Anime"]
    df = pd.DataFrame()
    for filename in os.listdir(directory):
        if((topics == 3) and filename[:-4] not in files):
            continue
        if((topics == 11) and (filename[:-4] == "3d_Printer")):
            continue
        if filename.endswith(".xml"): 
            if(itr == 0):
                df = get_df(directory+filename, filename, itr)
                print(df['Type'][0]+", "+str(len(df)))
            else:
                df1 = get_df(directory+filename, filename, itr)
                print(df1['Type'][0]+", "+str(len(df1)))
                df = pd.concat([df, df1], ignore_index=True, sort=False)
            itr = itr+1
        else:
            continue
    
    filename = "Combined_"+type_name+".csv"
    df.to_csv(filename, sep=',',index=False)
    return df

df = combine_dataset('Data/Training/', "Training_11_without_p", 3) #df.isna().sum()
# Audio(sound_file, autoplay=True)


# In[38]:


def train_test_split(documents):
    labels = documents['Label'].unique()
    
    train = pd.DataFrame()
    test = pd.DataFrame()
    validation = pd.DataFrame()
    
    for label in labels:
        temp = documents[documents['Label'] == label][:500]
        train = pd.concat([train, temp], ignore_index=True, sort=False)
        temp = documents[documents['Label'] == label][501:700]
        validation = pd.concat([validation, temp], ignore_index=True, sort=False)
        temp = documents[documents['Label'] == label][701:1200]
        test = pd.concat([test, temp], ignore_index=True, sort=False)
        
    trainX = train["sentences"]
    trainY = train["Label"]
    testX = test["sentences"]
    testY = test["Label"]
    valX = validation["sentences"]
    valY = validation["Label"]
    
    return trainX, testX, trainY, testY, valX, valY


# In[39]:


X_train, X_test, y_train, y_test, X_val, y_val = train_test_split(df)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
X_val = X_val.reset_index(drop=True)
y_val = y_val.reset_index(drop=True)



# # Hamming

# In[41]:


all_words = []
for item in df['sentences']:
    all_words += item
all_words = sorted(list(set(all_words)))

def hamming_distance(a, b):
    return np.count_nonzero(a!=b)


# # Euclidean

# In[42]:


def save_obj(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# In[43]:


def create_bow(documents, metric="Euclidean"): 
    bows = np.zeros((len(documents), len(all_words)))
    
    idx = 0
    
    if(metric == "Euclidean"):
      for sentence in tqdm(documents):
        for word in sentence:
          if(word in all_words):
            bows[idx][all_words.index(word)] += np.longdouble(1.0)
        idx += 1

    if(metric == "Hamming"):
      for sentence in tqdm(documents):
        for word in set(sentence):
          if(word in all_words):
            bows[idx][all_words.index(word)] = np.longdouble(1.0)
        idx += 1

    return bows


# In[44]:


def euclidean_distance(instance1, instance2):
    return np.longdouble(np.linalg.norm(np.longdouble(instance1) - np.longdouble(instance2)))


# # TF-IDF

# In[54]:


def generate_DF(X_train):
    DFreq_token = {}
    idx = 0
    for doc in tqdm(X_train):
        for token in doc:
            if token in DFreq_token:
                DFreq_token[token].append(idx)
            else:
                DFreq_token[token] = [idx]
        idx += 1

    for token in DFreq_token:
        DFreq_token[token] = np.longdouble(len(set(DFreq_token[token])))
    
    for token in all_words:
        if token in DFreq_token:
            continue
        else:
            DFreq_token[token] = np.longdouble(0)
    
    return DFreq_token

def get_DFreq_token(token, DFreq_token):
    if token in DFreq_token:
        N = np.longdouble(DFreq_token[token])
    else:
        N = np.longdouble(0)
    return N


# In[55]:


def generate_TF(doc):
    return dict((token, doc.count(token)) for token in set(doc))
        
def TF_IDF(doc, token, nd):
    d = generate_TF(doc)
    TF = np.longdouble(d[token])/np.longdouble(len(doc))
    IDF = np.log(np.longdouble(nd+1.0)/np.longdouble(get_DFreq_token(token, DF)+1.0))
    return TF*IDF

def cos_similarity(a, b):
    similarity = np.longdouble(1.0 * np.dot(a, b))/np.longdouble(1.0 * np.longdouble(np.linalg.norm(a))*np.longdouble(np.linalg.norm(b)))
    return similarity

def TF_IDFVectorize(X_train):
    ndoc = len(X_train)
    TF_IDF_Vector = np.zeros((ndoc, len(DF)))
    vocab = list(DF.keys())
    
    idx = 0
    for doc in tqdm(X_train):
        for token in doc:
            TF_IDF_Vector[idx][vocab.index(token)] = TF_IDF(doc, token, ndoc)
        idx += 1
    return TF_IDF_Vector


def get_TF_IDF(doc, ndoc):
    query = np.zeros((len(DF)))    
    vocab = list(DF.keys())

    for token in set(doc):
        idx = vocab.index(token)
        query[idx] = TF_IDF(doc, token, ndoc)
    return query


# In[56]:


### Use the stored pkl file (tf_idf.pkl)
DF = generate_DF(X_train)
TF_IDF_Vector = TF_IDFVectorize(X_train)


# # K-Nearest Neighbors

# In[47]:


import itertools

def fit_transform(X_train, X_test, X_val, topics, metric="Euclidean"):
    
    new_train = create_bow(X_train, metric)

    with open('new_train_'+metric+topics+'.pkl','wb') as f:
        pickle.dump(new_train, f)
    
    new_test = create_bow(X_test, metric)
    
    with open('new_test_'+metric+topics+'.pkl','wb') as f:
        pickle.dump(new_test, f)

    new_val = create_bow(X_val, metric)
    
    with open('new_val_'+metric+topics+'.pkl','wb') as f:
        pickle.dump(new_val, f)

    return new_train, new_test, new_val


def extract_fit_transform(metric, topics):
    with open('/content/drive/My Drive/ML_472/new_train_'+metric+topics+'.pkl','rb') as f:
        new_train = pickle.load(f)
        
    with open('/content/drive/My Drive/ML_472/new_test_'+metric+topics+'.pkl','rb') as f:
        new_test = pickle.load(f)
        
    with open('/content/drive/My Drive/ML_472/new_val_'+metric+topics+'.pkl','rb') as f:
        new_val = pickle.load(f)
        
    return new_train, new_test, new_val


# new_train, new_test, new_val = fit_transform(X_train, X_test, X_val, "11", "Hamming")
# new_train, new_test, new_val = fit_transform(X_train, X_test, X_val, "11", "Euclidean")
# new_train, new_test, new_val = extract_fit_transform("Euclidean", "11")
# Audio(sound_file, autoplay=True)
# del new_train, new_test, new_val


# In[57]:


def knn(X_train, Y_train, X_test, uniqueOutputCount, metric="TF-IDF", n_neighbors=3):
    allTestNeighbers=[]
    allPredictedOutputs =[]
    
    #calculate for earch test data points
    for testInput in X_test:
        allDistances = []
        
        if(metric == "Hamming"):
            for trainInput, trainActualOutput in zip(X_train, Y_train):
                distance = hamming_distance(testInput, trainInput)
                allDistances.append((trainInput, trainActualOutput, distance))

            #Sort (in ascending order) the training data points based on distances from the test point     

            allDistances.sort(key=lambda x: x[2])
        
        
        if(metric == "Euclidean"):
            for trainInput, trainActualOutput in zip(X_train, Y_train):
                distance = euclidean_distance(testInput, trainInput)
                allDistances.append((trainInput, trainActualOutput, distance))

            #Sort (in ascending order) the training data points based on distances from the test point     

            allDistances.sort(key=lambda x: x[2])
            
        if(metric == "TF-IDF"):
            testX = get_TF_IDF(testInput, len(X_train))
        
            for trainX, trainInput, trainActualOutput in zip(TF_IDF_Vector, X_train, Y_train):
                distance = cos_similarity(trainX, testX)
                allDistances.append((trainInput, trainActualOutput, distance))
        
            #Sort (in descending order) the training data points based on allignment to the test point     
        
            allDistances.sort(key=lambda x: x[2], reverse=True)
        
        #Assuming output labels are from 0 to uniqueOutputCount-1
        voteCount = np.zeros(uniqueOutputCount)
        neighbors = []
        for n in range(n_neighbors):
            neighbors.append(allDistances[n][0])
            class_label = int(allDistances[n][1])
            voteCount[class_label] += 1

        #Determine the Majority Voting (Equal weight considered)
        predictedOutput = np.argmax(voteCount)
        allTestNeighbers.append(neighbors)
        allPredictedOutputs.append(predictedOutput)
        
    return allPredictedOutputs, allTestNeighbers


# In[58]:


def performanceEvaluation(X_train, Y_train, X_test, Y_test, metric="TF-IDF", n_neighbors=3):
    totalCount = 0
    correctCount = 0

    #Determine Number of unique class lebels
    uniqueOutputLabels = []
    for label in Y_train:
        if label not in uniqueOutputLabels:
            uniqueOutputLabels.append(label)
    uniqueOutputCount = len(uniqueOutputLabels)
    
    for testInput, testActualOutput in tqdm(zip(X_test, Y_test), total=len(X_test)):
        predictedOutput,_ = knn(X_train, Y_train, [testInput], uniqueOutputCount, metric, n_neighbors)
        
        if predictedOutput[0] == testActualOutput:
            correctCount += 1
        totalCount += 1
    
    accuracy = np.longdouble(correctCount*100)/np.longdouble(totalCount)
    print("Total Correct Count: ",correctCount," Total Wrong Count: ",totalCount-correctCount," Accuracy: ",accuracy)
    return correctCount, accuracy
    

def performanceValidation(X_train, Y_train, X_test, Y_test, X_val, Y_val, metric="TF-IDF"):

    #Determine Number of unique class lebels
    uniqueOutputLabels = []
    for label in Y_train:
        if label not in uniqueOutputLabels:
            uniqueOutputLabels.append(label)
    uniqueOutputCount = len(uniqueOutputLabels)
    
    n_neighbors = [1, 3 , 5]
    
    accuracy_list = []
    
    output_file = open('Output_KNN.txt', 'a', encoding='utf-8')
    
    output_file.write("Method: %s\n" % (metric))
    
    for k in n_neighbors:
        totalCount = 0
        correctCount = 0
        
        for testInput, testActualOutput in tqdm(zip(X_val, Y_val), total=len(X_val)):
            predictedOutput,_ = knn(X_train, Y_train, [testInput], uniqueOutputCount, metric, k)
            
            if predictedOutput[0] == testActualOutput:
                correctCount += 1
            totalCount += 1
        
        print("For k = ", k, ", Total Correct Count : ",correctCount," Total Wrong Count: ",totalCount-correctCount," Accuracy: ",(correctCount*100)/(totalCount))
        output_file.write("For k = %d, Total Correct Count : %d  Total Wrong Count: %d  Accuracy: %f\n" % (k, correctCount, totalCount-correctCount, (correctCount*100)/(totalCount)))
        accuracy_list.append(np.longdouble(correctCount*100)/np.longdouble(totalCount))
        
    print(accuracy_list)
    
    best_k = n_neighbors[accuracy_list.index(max(accuracy_list))]
    print("Best k = ", best_k)
    output_file.write("\nBest k = %d\n\n" % (best_k))
    
    test_correctCount, test_accuracy = performanceEvaluation(X_train, Y_train, X_test, Y_test, metric, best_k)
    output_file.write("Total Correct Count: %d  Total Wrong Count: %d  Accuracy: %f\n\n" % (test_correctCount, len(X_test) - test_correctCount, test_accuracy))
    output_file.close()
    
    return best_k, test_correctCount, test_accuracy


# best_k, test_correctCount, test_accuracy = performanceValidation(new_train, y_train, new_test, y_test, new_val, y_val, "Hamming")
# best_k, test_correctCount, test_accuracy = performanceValidation(new_train, y_train, new_test, y_test, new_val, y_val, "Euclidean")    
best_k, test_correctCount, test_accuracy = performanceValidation(X_train, y_train, X_test, y_test, X_val, y_val, "TF-IDF")  


# In[50]:


def run_KNNclassifier(topics):
    
    method = "TF-IDF"
    best_k, test_correctCount, test_accuracy = performanceValidation(X_train, y_train, X_test, y_test, X_val, y_val, method) 
    
    method = "Hamming"
    new_train, new_test, new_val = fit_transform(X_train, X_test, X_val, topics, method)
    best_k, test_correctCount, test_accuracy = performanceValidation(new_train, y_train, new_test, y_test, new_val, y_val, method) 
    
    method = "Euclidean"
    new_train, new_test, new_val = fit_transform(X_train, X_test, X_val, topics, method)
    best_k, test_correctCount, test_accuracy = performanceValidation(new_train, y_train, new_test, y_test, new_val, y_val, method) 
    
    return 

run_KNNclassifier("11")


# In[ ]:


# performanceEvaluation(new_train, y_train, new_test, y_test, "Hamming", 3)
performanceEvaluation(new_train, y_train, new_test, y_test, "Eucledian", 1)
# performanceEvaluation(X_train, y_train, X_test, y_test, "TF-IDF", 1)     



# # Naive Bayes

# In[12]:


uniqueOutputLabels = []
for label in y_train:
    if label not in uniqueOutputLabels:
        uniqueOutputLabels.append(label)
uniqueOutputCount = len(uniqueOutputLabels)

all_training_words = []
for item in X_train:
    all_training_words += item

all_training_words = sorted(list(set(all_training_words)))


# In[13]:


def get_typeInfo(Y_train):
    posts_per_type = {}
    priors = {}
    
    temp = pd.Index(Y_train)
    
    for label in uniqueOutputLabels:
        posts_per_type[label] = np.longdouble(temp.value_counts()[label])
        priors[label] = np.longdouble(posts_per_type[label])/np.longdouble(len(Y_train))
    
    return posts_per_type, priors


def likelihood(X_train, y_train):
    nWords_per_class = {}
    wordCount_per_class = {}

    for label in np.unique(y_train):
        wordCount_per_class[label] = {}
        
    for doc, label in zip(X_train, y_train):
        for token in doc:
            try:
                wordCount_per_class[label][token] += np.longdouble(1)
            except:
                wordCount_per_class[label][token] = np.longdouble(1)
            if label in nWords_per_class:
                nWords_per_class[label] += np.longdouble(1)
            else:
                nWords_per_class[label] = np.longdouble(1)
    return wordCount_per_class, nWords_per_class


# In[14]:


posts_per_type, priors = get_typeInfo(y_train)
wordCount_per_class, nWords_per_class = likelihood(X_train, y_train)


# In[17]:


def get_Nwc(c, token, wordCount_per_class):
    try:
        N_wc = wordCount_per_class[c][token] 
    except:
        N_wc = np.longdouble(0)
    return N_wc


def NB(X_test, Y_test, alpha=1):

    accurate = 0
    totalCount = 0
    
    for doc, label in zip(X_test, Y_test):
        likelihood_per_label = {}
        for token in set(doc):
            for c in uniqueOutputLabels:
                if c in likelihood_per_label:
                    likelihood_per_label[c] += np.log(np.longdouble(get_Nwc(c, token, wordCount_per_class) + alpha)/np.longdouble(alpha*np.longdouble(nWords_per_class[c])))
                else:
                    likelihood_per_label[c] = np.log(np.longdouble(priors[c])) + np.log(np.longdouble(get_Nwc(c, token, wordCount_per_class) + alpha)/np.longdouble(alpha*np.longdouble(nWords_per_class[c])))

        
        likelihood_per_label = {k: v for k, v in sorted(likelihood_per_label.items(), key=lambda item: item[1], reverse=True)}
        
        if(list(likelihood_per_label.keys())[0] == label):
            accurate += 1
        totalCount += 1
    
    accuracy = np.longdouble(accurate*100)/np.longdouble(totalCount)
    print("For alpha = ", alpha, ", Total Correct Count: ",accurate," Total Wrong Count: ",totalCount-accurate," Accuracy: ", accuracy)        
    return accuracy, accurate

    
NB(X_test, y_test, 0.02)


# In[19]:


def NB_Validation(X_train, Y_train, X_test, Y_test, X_val, Y_val):
    
    smoothing_factors = np.random.uniform(low=0.01, high=0.05, size=(10,))
    
    accuracy_list = []
    idx = 0
    for sm in smoothing_factors:
        print("Iter ", idx, ":")
        accuracy, n_accurate = NB(X_val, Y_val, np.longdouble(sm))
        accuracy_list.append([accuracy, sm])
        idx += 1
        
    
    accuracy_list.sort(key = lambda x: x[0], reverse=True) 
    
    best_sf = accuracy_list[0][1]
    
    print("\nBest Smoothing Factor: ",best_sf)
    accuracy, n_accurate = NB(X_test, Y_test, best_sf)
    
    return accuracy, n_accurate
        
accuracy, n_accurate = NB_Validation(X_train, y_train, X_test, y_test, X_val, y_val)



# In[72]:


from pandas.util.testing import assert_frame_equal

def prepare_testSet(temp_df):
    
    labels = temp_df['label'].unique()
    idx = 0
    
    print(len(labels))
    sub_test = pd.DataFrame(columns=['texts', 'label'])
    
    for label in labels:
        temp = temp_df[temp_df['label'] == label]
        indices = temp.index.values.tolist()
        temp_df = temp_df.drop(indices[0:10])
        temp = temp.reset_index(drop=True)
        if(idx == 0):
            sub_test = temp[0:10]
        else:
            sub_test = sub_test.append(temp[0:10], ignore_index=True)
        idx += 1
    
    return sub_test, temp_df


temp_df = pd.DataFrame()
temp_df['texts'] = X_test
temp_df['label'] = y_test


results_test_nb = []
results_test_knn = []

for i in range(50):
    sub_test, temp_df = prepare_testSet(temp_df)    
    accuracy, n_accurate = NB(sub_test['texts'], sub_test['label'], 0.02167009955202075)
    results_test_nb.append(accuracy)
    print(accuracy)
    
    test_correctCount, test_accuracy = performanceEvaluation(X_train, y_train, sub_test['texts'], sub_test['label'], "TF-IDF", 5)
    results_test_knn.append(test_accuracy)  
    print(test_accuracy)


# In[93]:


def t_stat(results_test_nb, results_test_knn):
    a = np.asarray(results_test_nb, dtype=np.longdouble)
    b = np.asarray(results_test_knn, dtype=np.longdouble)
    # compare samples
    significance_levels = [0.005, 0.01, 0.05]

    stat, p = ttest_rel(a, b)
    print('Statistics=%.3f, p=%lf\n' % (stat, p))



    print("For significance level 0.05: ")
    if(p < 0.05):
        print("Reject null hypothesis that the means are equal\n")
    else:
        print("Accept null hypothesis that the means are equal\n")

    print("For significance level 0.01: ")
    if(p < 0.01):
        print("Reject null hypothesis that the means are equal\n")
    else:
        print("Accept null hypothesis that the means are equal\n")

    print("For significance level 0.005: ")
    if(p < 0.005):
        print("Reject null hypothesis that the means are equal\n")
    else:
        print("Accept null hypothesis that the means are equal\n")
        
t_stat(results_test_nb, results_test_knn)

