import pandas as pd
import numpy as np


# CSG: calling strength graph
# preds: prediction of methods' labels
# src: the class where the method locates
# Note that, as CSG is a sparse graph, non-zero elements is about 2% of the total element.
# For activemq, the number of method is about 5E4, number of iterations exceeded 2.5E9.
# Traversing sparse matrix is as faster as 2500 times than traversing dense matrix.
def refactoring(CSG, preds, src):
    idx = CSG._indices()
    data = CSG._values()

    source_class = []
    smelly_method = []
    target_class = []

    data_i = 0
    while data_i < len(data):
        caller_id = idx[0][data_i]
        caller_source_class = src[caller_id]
        if preds[caller_id] == 1:             # the method is predicted to be smelly
            CS = np.full(max(src)+1, -np.inf) # calling strength between method and class
            while data_i < len(data) and idx[0][data_i] == caller_id:
                callee_id = idx[1][data_i]
                callee_source_class = src[callee_id]
                if np.isinf(CS[callee_source_class]):
                    CS[callee_source_class] = data[data_i]
                else:
                    CS[callee_source_class] += data[data_i]
                data_i += 1
            source_class.append(caller_source_class)
            smelly_method.append(caller_id.cpu().numpy().max())
            target_class.append(np.argmax(CS))
        else:
            data_i += 1
    return source_class, smelly_method, target_class

def saveRefactoringResults(project, preds, calling_strength):
    df1 = pd.read_csv('data/' + project + '/method.csv')
    df2 = pd.read_csv('data/' + project + '/classInfo.csv')

    method_locate_class = df1['classID'].values

    res = pd.DataFrame()
    source_class, smelly_method, target_class = refactoring(CSG=calling_strength, preds=preds, src=method_locate_class)

    res['sourceClass'] = df2['className'][source_class].values
    res['method'] = df1['completename'][smelly_method].values
    res['targetClass'] = df2['className'][target_class].values

    res.to_csv('results/'+project+'_refactoring.csv', index=False)

    calculate_Acc(project)

def calculate_Acc(project):
    ground_truth = pd.read_csv('data/'+project+'/ground_truth.csv')
    pred_result = pd.read_csv('results/'+project+'_refactoring.csv')
       
    method_1 = pred_result['method'].values.tolist()
    pred_targetClass = pred_result['targetClass'].values.tolist()
    
    method_2 = ground_truth['method'].values.tolist()
    targetClass = ground_truth['targetClass'].values.tolist()

    correct_TP = 0
    TP = 0
    for i in range(0, len(method_1)):
        for j in range(0, len(method_2)):
            if method_1[i] == method_2[j]:
                TP += 1
                if pred_targetClass[i] == targetClass[j]:
                    correct_TP += 1
    
    print("refactoring accuracy:%.2f"%(100.0*correct_TP/TP))