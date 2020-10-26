import numpy as np
import pandas as pd

# Reading the datasets
wine_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
ttt_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data'
wine_attributes = ['Class','Alcohol','Malic acid','Ash length','Alcalinity of ash',\
                   'Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols',\
                    'Proanthocyanins','Color intensity','Hue','OD280','Proline']
#Wine dataset NOTE: 1st attribute is class identifier (1-3)
wine_df = pd.read_csv(wine_URL, names =wine_attributes)
#Rearrange wine dataframe
cols = wine_df.columns.tolist() 
wine_df = wine_df[cols[1:]+cols[:1]]

tic_tak_atts = {'0','1','2', '3', '4','5','6','7','8','9'}
ttt_df = pd.read_csv(ttt_URL,names= tic_tak_atts)


# Sample Small Dataset for checking the performance of functions
outlook = 'overcast,overcast,overcast,overcast,rainy,rainy,rainy,rainy,rainy,sunny,sunny,sunny,sunny,sunny'.split(',')
temp = 'hot,cool,mild,hot,mild,cool,cool,mild,mild,hot,hot,mild,cool,mild'.split(',')
humidity = 'high,normal,high,normal,high,normal,normal,normal,high,high,high,high,normal,normal'.split(',')
windy = 'FALSE,TRUE,TRUE,FALSE,FALSE,FALSE,TRUE,FALSE,TRUE,FALSE,TRUE,FALSE,FALSE,TRUE'.split(',')
play = 'yes,yes,yes,yes,yes,yes,no,yes,no,no,no,no,yes,yes'.split(',')

dataset ={'outlook':outlook,'temp':temp,'humidity':humidity,'windy':windy,'play':play}
df = pd.DataFrame(dataset,columns=['outlook','temp','humidity','windy','play'])


def Entropy(target_attribute):
    X , counts = np.unique(target_attribute, return_counts = True)
    pX = [ci/len(target_attribute) for ci in counts]
    entropy = -1*(pX@np.log2(pX))
    return entropy

def conditional_Entropy(Y,given): #H(Y|given)
    result =  0
    values, counts = np.unique(given, return_counts = True)
    for v , c  in zip(values,counts):
        splited_Y = [Y[i] for i in range(len(Y)) if given[i]==v]
        H = Entropy(splited_Y)
        p = c/len(given)
        result += p*H  #-1*sum(p*H)
    return result

def Gain(X,given):
    return Entropy(X) - conditional_Entropy(X,given)



class TNode:
    def __init__(self,name,parent=None,par_val=None):
        self.name= name
        self.parent = parent
        self.children = [] #list of TNodes, if empty =>this node is a leaf
        self.values = []
        self.par_val = par_val
        
 
    def set_values(self,v):
        self.values = list(v)
        
        
        
def find_winner(DataFrame):
    features = list(DataFrame.columns)
    feature_dic = {features[i]:Gain(list(DataFrame[features[i]]),\
                   list(DataFrame[features[-1]]))\
                for i in range(0,len(features)-1)}

    max_gain = max(feature_dic.values())
    index_max = list(feature_dic.values()).index(max_gain)
    winner_name = list(feature_dic.keys())[index_max]
    
    winner_node = TNode(winner_name)
    branches = np.unique(DataFrame[winner_name])
    winner_node.set_values(branches)
    return winner_node

def most_frequent(List): 
    return max(set(List), key = List.count)
               
def ID3(df,depth=0, root=None):
    if root==None: #Only the first time
        #print("Building ID3...")
        root = find_winner(df)
        #print('** Root=  ', root.name,'\n\n')
    if root.parent:
        if len(root.values)<len(root.parent.values):
            depth =20
    depth +=1
    if depth < 20:# To avoid maximum recursion error, and also over-fitting              
        for att in root.values:
            child_df = df[df[root.name] == att]#Spliting dataset
            child_df.drop(root.name,axis=1) #Removing the selected attribute
            remained_class_labels = list(child_df[list(child_df.columns)[-1]])
            child_labels = np.unique(remained_class_labels) # Remained Labels
            if len(child_labels)>1:#not leaf-> recurcive
                next_winner = find_winner(child_df)
                next_winner.parent = root
                next_winner.par_val=att
                root.children.append(next_winner)
                ID3(child_df,depth, root=next_winner)
        
            else:#else if we reached a leaf
                leaf = TNode(child_labels[0])
                leaf.parent=root
                leaf.par_val=att
                root.children.append(leaf)
                depth =1
    else:
        #print('happend')
        common_label = most_frequent(list(df[list(df.columns)[-1]]))
        leaf =TNode(common_label)
        root.children = []
        root.values = []
        leaf.parent=root.parent
        leaf.par_val=root.par_val
        root.parent.children.remove(root)
        root.parent.children.append(leaf)
        depth =1

    return root
            
            
            

def rep_tree(root,depth=-1):
    if len(root.values)==0:#leaf
        print('|',root.name,'|')
        return
    else:
        depth +=1
        for child in root.children:
            print(depth*'\t',root.name,'=',child.par_val,'->')
            rep_tree(child,depth)
            
            
#print('Play Tennis?') 
#print(rep_tree(ID3(df)))
#
#test = ["rainy","hot","normal","TRUE"]

    
def ttt_pred(root,keys,test,lables=None):
    try:
        if len(root.children)==0:
            return root.name
        if root.values[0]==test[keys.index(root.name)]:
            return ttt_pred(root.children[0],keys,test,lables)
        if root.values[1]==test[keys.index(root.name)]:
            return ttt_pred(root.children[1],keys,test,lables)
        if root.values[2]==test[keys.index(root.name)]:
            return ttt_pred(root.children[2],keys,test,lables)
    except:
        print('!', end= ' ')
        return lables[np.random.randint(0,high=len(lables))]

def wine_pred(root,keys,test,lables=None):
    try:
        if len(root.children)==0:
            return root.name
        elif root.values[0] == test[keys.index(root.name)]:
            return wine_pred(root.children[0],keys,test,lables)
        else:# root.values[1] == test[keys.index(root.name)]:
            return wine_pred(root.children[1],keys,test,lables)
    except:
        print('e', end =' ' )
        return lables[np.random.randint(0,high=len(lables))]

       
def binary_confusion(GT,y_predicted):
    TP,FP,TN,FN = 0,0,0,0
    for index in range(len(GT)):
        if GT[index]=='positive':
            if y_predicted[index]=='positive':
                TP +=1
            else:
                FP +=1
        else:
            if y_predicted[index]=='negative':
                TN +=1
            else:
                FN +=1     
    return[[TP,FP],[FN,TN]]
                
    
'''10 times * 10-Fold Cross Validation'''            
#------------ Tic-Tak-Toe
Accuracy,confusion = [] ,[]  
d = int(len(ttt_df)/10)+1
ttt_lables = np.unique(list(ttt_df[ttt_df.columns[-1]]))
for times in np.arange(10):
    Dataset = ttt_df.sample(frac=1)#shuffling dataset
    
    Folds = [Dataset[:d],\
             Dataset[d:2*d],Dataset[2*d:3*d],Dataset[3*d:4*d],\
             Dataset[4*d:5*d],Dataset[5*d:6*d],Dataset[6*d:7*d],\
             Dataset[7*d:8*d],Dataset[8*d:9*d],\
             Dataset[9*d:]
             ]
    for i in range(0,10):#10 Folds cross-validation
        '''Train ID3 based on the 9-Folds and Test on 1 Fold '''
        test_df = Folds[i]
        train_df = pd.concat([Folds[i-9],Folds[i-8],Folds[i-7],\
                              Folds[i-6],Folds[i-5],Folds[i-4],\
                              Folds[i-3],Folds[i-2],Folds[i-1]])
        root = ID3(train_df) 
        GT = test_df[test_df.columns[-1]].to_list()
        test_df = test_df.drop(test_df.columns[-1],axis=1)#Romove the labels
        
        y_pred = []
        for r in np.arange(len(test_df)):
            test = test_df.iloc[r,:].to_list()
            prediction = ttt_pred(root,list(test_df.columns),test,\
                                  lables=ttt_lables)
            y_pred.append(prediction)
            
        conf = binary_confusion(GT,y_pred) 
        confusion.append(conf)
        Accuracy.append((conf[0][0]+conf[1][1])/len(GT))

print('Tic-Tak_Toe')
print('\tMean(Accuracy)=%{:.2f}, Var(Accuracy)= {:.4f}'\
      .format(np.mean(Accuracy)*100,np.var(Accuracy)))

print('\tBest Confusion Matrix = ', confusion[np.argmax(Accuracy)])           
        

#------------ Wine dataset

# 1. binary split using a threshold 
for column in wine_df.columns:
    if column == 'Class':
        break
    att = wine_df[column].to_list()
    threshhold = np.median(att)
    binary_col = [True if a>threshhold else False for a in att]
    wine_df[column] = binary_col
     
def confusion_mat(GT,predicted,lables):
    count11,count12,count13 = 0,0,0
    count21,count22,count23 = 0,0,0
    count31,count32,count33 = 0,0,0
    
    for i in range(len(GT)):
        if GT[i]==lables[0]:
            if predicted[i] ==lables[0]:
                count11 +=1
            elif predicted[i] ==lables[1]:
                count12 +=1
            elif predicted[i] ==lables[2]:
                count13 +=1
        elif GT[i]==lables[1]:
            if predicted[i] ==lables[0]:
                count21 +=1
            elif predicted[i] ==lables[1]:
                count22 +=1
            elif predicted[i] ==lables[2]:
                count23 +=1
        elif GT[i]==lables[2]:
            if predicted[i] ==lables[0]:
                count31 +=1
            elif predicted[i] ==lables[1]:
                count32 +=1
            elif predicted[i] ==lables[2]:
                count33 +=1
    return [[count11,count12,count13],\
            [count21,count22,count23],[count31,count32,count33]]

#--- 10 times * 10-Fold Cross Validation            
wine_Accuracy,wine_confusion = [] ,[]  
d = int(len(Dataset)/10)+1
wine_lables  =np.unique(list(wine_df.Class))
for times in range(0,10):
    Dataset = wine_df.sample(frac=1)#shuffling dataset
    Folds = [Dataset[:d],\
             Dataset[d:2*d],Dataset[2*d:3*d],Dataset[3*d:4*d],\
             Dataset[4*d:5*d],Dataset[5*d:6*d],Dataset[6*d:7*d],\
             Dataset[7*d:8*d],Dataset[8*d:9*d],\
             Dataset[9*d:]
             ]
        #10 Folds cross-validation
        #Train ID3 based on the 9-Folds, then Test on 1 Fold 
    for i in np.arange(10):
        test_df = Folds[i]
        train_df = pd.concat([Folds[i-9],Folds[i-8],Folds[i-7],\
                              Folds[i-6],Folds[i-5],Folds[i-4],\
                              Folds[i-3],Folds[i-2],Folds[i-1]])
        root = ID3(train_df) 
        y_true = test_df[test_df.columns[-1]].to_list()
        y_wine_pred = []
        for r in range(0,len(test_df)):
            test = test_df.iloc[r,:].to_list()
            prediction = wine_pred(root,list(test_df.columns),test,\
                                   lables=wine_lables)
            y_wine_pred.append(prediction)
        #test_df = test_df.drop(test_df.columns[-1],axis=1)#Romove the labels
#        tests = [test_df.iloc[i,:].to_list() for i in range(len(test_df))]
#        y_pred = [wine_pred(root,list(test_df.columns),t,lables=wine_labels)\
#                  for t in tests]
        conf = confusion_mat(y_true,y_wine_pred,wine_lables)
        if len(y_true)>0:
            wine_confusion.append(conf)
            wine_Accuracy.append((conf[0][0]+conf[1][1]+conf[2][2])/len(y_true))

print('Wine Dataset:')
print('\tMean(Accuracy)=%{:.2f}, Var(Accuracy)= {:.4f}'\
      .format(np.mean(wine_Accuracy)*100,np.var(wine_Accuracy)))
print('\tBest Confusion Matrix =\n ', wine_confusion[np.argmax(wine_Accuracy)])           
          

#--------------------------------------------------------
    #b
#--------------------------------------------------------
        
        
def gain_ratio(X,given):
    values, counts = np.unique(given, return_counts = True)
    splitInfo = 0
    for c  in counts:
        p = c/len(given)
        splitInfo -= p*np.log2(p)  
    return Gain(X,given)/ splitInfo      
        

        
def GainRatio_winner(DataFrame):
    features = list(DataFrame.columns)
    feature_dic = {features[i]:gain_ratio(list(DataFrame[features[i]]),\
                   list(DataFrame[features[-1]]))\
                for i in range(0,len(features)-1)}

    max_gain = max(feature_dic.values())
    index_max = list(feature_dic.values()).index(max_gain)
    winner_name = list(feature_dic.keys())[index_max]
    
    winner_node = TNode(winner_name)
    branches = np.unique(DataFrame[winner_name])
    winner_node.set_values(branches)
    return winner_node

               
def ID3_GR(df,root=None,depth=0):
    if root==None: #Only the first time
        #print("Building ID3...")
        root = GainRatio_winner(df)
        #print('** Root=  ', root.name,'\n\n')
    if root.parent:
        if len(root.values)<len(root.parent.values):
            depth =20
    depth +=1
    if depth < 20:# To avoid maximum recursion error and also over-fitting              
        for att in root.values:
            child_df = df[df[root.name] == att]#Spliting dataset
            child_df.drop(root.name,axis=1) #Removing the selected attribute
            remained_class_labels = list(child_df[list(child_df.columns)[-1]])
            child_labels = np.unique(remained_class_labels) # Remained Labels
            if len(child_labels)>1:#not leaf-> recurcive
                next_winner = GainRatio_winner(child_df)
                next_winner.parent = root
                next_winner.par_val=att
                root.children.append(next_winner)
                ID3(child_df,depth, root=next_winner)
        
            else:#else if we reached a leaf
                leaf = TNode(child_labels[0])
                leaf.parent=root
                leaf.par_val=att
                root.children.append(leaf)
                depth =1
    else:
        #print('happend')
        common_label = most_frequent(list(df[list(df.columns)[-1]]))
        leaf =TNode(common_label)
        root.children = []
        root.values = []
        leaf.parent=root.parent
        leaf.par_val=root.par_val
        root.parent.children.remove(root)
        root.parent.children.append(leaf)
        depth =1

    return root
            
           
'''10 times * 10-Fold Cross Validation'''            
#------------ Tic-Tak-Toe
GRAccuracy,GRconfusion = [] ,[]  
d = int(len(ttt_df)/10)+1
ttt_lables = np.unique(list(ttt_df[ttt_df.columns[-1]]))
for times in range(10):
    Dataset = ttt_df.sample(frac=1)#shuffling dataset
    
    Folds = [Dataset[:d],\
             Dataset[d:2*d],Dataset[2*d:3*d],Dataset[3*d:4*d],\
             Dataset[4*d:5*d],Dataset[5*d:6*d],Dataset[6*d:7*d],\
             Dataset[7*d:8*d],Dataset[8*d:9*d],\
             Dataset[9*d:]
             ]
    for i in range(0,10):#10 Folds cross-validation
        '''Train ID3 based on the 9-Folds and Test on 1 Fold '''
        test_df = Folds[i]
        train_df = pd.concat([Folds[i-9],Folds[i-8],Folds[i-7],\
                              Folds[i-6],Folds[i-5],Folds[i-4],\
                              Folds[i-3],Folds[i-2],Folds[i-1]])
        #    print("test df:\n", test_df)
        #    print('train_df:\n', train_df,'\n')
        root = ID3_GR(train_df) 
        GT = test_df[test_df.columns[-1]].to_list()
        test_df = test_df.drop(test_df.columns[-1],axis=1)#Romove the labels
        
        y_pred = []
        for row in range(0,len(test_df)):
            test = test_df.iloc[row,:].to_list()
            prediction = ttt_pred(root,list(test_df.columns),test,lables=ttt_lables)
            y_pred.append(prediction)
            
        conf = binary_confusion(GT,y_pred) 
        GRconfusion.append(conf)
        GRAccuracy.append((conf[0][0]+conf[1][1])/len(GT))

print('Tic-Tak_Toe with Gain Ratio')
print('\tMean(Accuracy)=%{:.2f}, Var(Accuracy)= {:.4f}'\
      .format(np.mean(GRAccuracy)*100,np.var(GRAccuracy)))

print('\tBest Confusion Matrix = ', confusion[np.argmax(GRAccuracy)])           
        

#------------ Wine dataset
#--- 10 times * 10-Fold Cross Validation            
GRwine_Accuracy,GRwine_confusion = [] ,[]  
d = int(len(Dataset)/10)+1
wine_lables  =np.unique(list(wine_df.Class))
for times in range(0,10):
    Dataset = wine_df.sample(frac=1)#shuffling dataset
    Folds = [Dataset[:d],\
             Dataset[d:2*d],Dataset[2*d:3*d],Dataset[3*d:4*d],\
             Dataset[4*d:5*d],Dataset[5*d:6*d],Dataset[6*d:7*d],\
             Dataset[7*d:8*d],Dataset[8*d:9*d],\
             Dataset[9*d:]
             ]
        #10 Folds cross-validation
        #Train ID3 based on the 9-Folds, then Test on 1 Fold 
    for i in range(0,10):
        test_df = Folds[i]
        train_df = pd.concat([Folds[i-9],Folds[i-8],Folds[i-7],\
                              Folds[i-6],Folds[i-5],Folds[i-4],\
                              Folds[i-3],Folds[i-2],Folds[i-1]])
        root = ID3_GR(train_df) 
        y_true = test_df[test_df.columns[-1]].to_list()
        y_wine_pred = []
        for i in range(0,len(test_df)):
            test = test_df.iloc[i,:].to_list()
            prediction = wine_pred(root,list(test_df.columns),test,lables=wine_lables)
            y_wine_pred.append(prediction)
        #test_df = test_df.drop(test_df.columns[-1],axis=1)#Romove the labels
#        tests = [test_df.iloc[i,:].to_list() for i in range(len(test_df))]
#        y_pred = [wine_pred(root,list(test_df.columns),t,lables=wine_labels)\
#                  for t in tests]
        conf = confusion_mat(y_true,y_wine_pred,wine_lables)
        if len(y_true)>0:
            GRwine_confusion.append(conf)
            GRwine_Accuracy.append((conf[0][0]+conf[1][1]+conf[2][2])/len(y_true))

print('Wine Dataset with Gain Ratio:')
print('\tMean(Accuracy)=%{:.2f}, Var(Accuracy)= {:.4f}'\
      .format(np.mean(GRwine_Accuracy)*100,np.var(GRwine_Accuracy)))
print('\tBest Confusion Matrix =\n\t', wine_confusion[np.argmax(GRwine_Accuracy)])           
          
        
        
        
        
        
        
        
        
        