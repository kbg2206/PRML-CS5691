#!/usr/bin/env python
# coding: utf-8

# # General Instructions to students:
# 
# 1. There are 4 types of cells in this notebook. The cell type will be indicated within the cell.
#     1. Markdown cells with problem written in it. (DO NOT TOUCH THESE CELLS) (**Cell type: TextRead**)
#     2. Python cells with setup code for further evaluations. (DO NOT TOUCH THESE CELLS) (**Cell type: CodeRead**)
#     3. Python code cells with some template code or empty cell. (FILL CODE IN THESE CELLS BASED ON INSTRUCTIONS IN CURRENT AND PREVIOUS CELLS) (**Cell type: CodeWrite**)
#     4. Markdown cells where a written reasoning or conclusion is expected. (WRITE SENTENCES IN THESE CELLS) (**Cell type: TextWrite**)
#     
# 2. You are not allowed to insert new cells in the submitted notebook.
# 
# 3. You are not allowed to import any extra packages, unless needed.
# 
# 4. The code is to be written in Python 3.x syntax. Latest versions of other packages maybe assumed.
# 
# 5. In CodeWrite Cells, the only outputs to be given are plots asked in the question. Nothing else to be output/printed.
# 
# 6. If TextWrite cells ask you to give accuracy/error/other numbers, you can print them on the code cells, but remove the print statements before submitting.
# 
# 7. Any runtime failures on the submitted notebook will get zero marks.
# 
# 8. All code must be written by you. Copying from other students/material on the web is strictly prohibited. Any violations will result in zero marks.
# 
# 10. All plots must be labelled properly, the labels/legends should be readable, all tables must have rows and columns named properly.
# 
# 11. Change the name of file with your roll no. For example cs15d203.ipynb (for notebook) and cs15d203.py (for plain python script)
# 
# 
# 
# 

# In[ ]:


# Cell type : CodeRead

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 
# **Cell type : TextRead**
# 
# 4b) Write a code to do polynomial regression with quadratic regularization that takes degree d and regularization parameter λ as input.
# 
# You could refer to the linear regression code discussed in [Tutorial 8](https://colab.research.google.com/drive/1kQd5F0dDFFRnyduG5uB7UWHtyFQJjMHq?usp=sharing#scrollTo=dqri9wcqb-k7).

# In[ ]:


# Cell type : CodeWrite
# write the function for Polynomial regression with quadratic regularization here.

Data=pd.read_csv("bayes_variance_data.csv")
x = Data.iloc[:, :-1].values
y = Data.iloc[:, -1].values
def designMatrix(data,degree):
    
    matrix=np.zeros((len(data),degree+1))
    for i in range(degree+1):
        matrix[:,i]=data[:,0]**i
        
    return matrix
        

def polyregression(x,y,lambda_l,degree):
    
    
    basis_function=designMatrix(x,degree)
    I = np.identity(degree+1)
    weights = np.linalg.inv(basis_function.T @ basis_function + lambda_l*I) @ basis_function.T @ y
    
    return weights
    
    
    
polyregression(x,y,0.1,24)




# 4c) Run the code for degree $d=24$ and each $\lambda$ in the set:
# \[\{10^{-15}, 10^{-9}, 10^{-6}, 10^{-3}, 10^{-2}, 10^{-1}, 1, 10^{1}, 10^{2}, 10^{3}, 10^{6}, 10^{9}, 10^{15}\}\]
# 
#   i) Perform 5-fold cross-validation on the 100 data points (20 datapoints in each fold). For each fold, compute both training and validation errors using the mean squared error loss function. \\
#   ii) Calculate the average training and validation errors across the 5 folds.

# In[ ]:


def Min_Square_Error(y_new,y_old):
    total=0
    for i in range(len(y_new)):
        total+=(y_new[i]-y_old[i])**2
    return total/len(y_new)

def KFold_Poly(Data,lambda_l):
    
    x = Data.iloc[:, :-1].values
    y = Data.iloc[:, -1].values
    
   
    Training_score = []
    Validation_score = []
    degree=24
    fold_size=20
    
    for i in range(5):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        x_test_set = x[test_start:test_end]
        y_test_set = y[test_start:test_end]
        
        if i == 0:
            x_train_set = x[test_end:]
            y_train_set = y[test_end:]
            
        elif i == 4:
            x_train_set = x[:test_start]
            y_train_set = y[:test_start]
        else:
            x_train_set = np.concatenate((x[:test_start], x[test_end:]))
            y_train_set = np.concatenate((y[:test_start], y[test_end:]))
            
        weights=polyregression(x_train_set,y_train_set,lambda_l,degree)
        
        #print(type(x_train_set))
        x_train_set=designMatrix(x_train_set,degree)
        x_test_set=designMatrix(x_test_set,degree)
             
        y_pred_training=np.dot(x_train_set,weights)
        y_pred_validation=np.dot(x_test_set,weights)
       
        training_mse=Min_Square_Error(y_pred_training,y_train_set)
        validation_mse=Min_Square_Error(y_pred_validation,y_test_set)
       
    
        
        Training_score.append(training_mse)  
        Validation_score.append(validation_mse)
            
            
            
    
    return Training_score, Validation_score

KFold_Poly(Data,0.1)


# 4d)  Construct a learning curve by plotting the average training and validation errors against the model complexity ($\log_{10} \lambda$). Based on this learning curve, identify the (i) model with the highest bias, (ii) model with the highest variance?, and (iii) the model that will work best on some unseen data.

# In[ ]:


def plot_learning_curve(lambda_values, avg_training_errors, avg_validation_errors):
    plt.figure(figsize=(10, 6))
    plt.plot(np.log10(lambda_values), avg_training_errors,color='Black', marker='o', label='Training Error')
    plt.plot(np.log10(lambda_values), avg_validation_errors,color='Red', marker='^', label='Validation Error')
    
    
    min_val_error = min(avg_validation_errors)
    min_val_lambda = lambda_values[avg_validation_errors.index(min_val_error)]
    plt.axvline(np.log10(min_val_lambda), color='Orange', linestyle='--', label='Best Model')
    
    High_bias_point=max(lambda_values)
    plt.axvline(np.log10(High_bias_point), color='Green', linestyle='--', label='High Bias')
    
    High_Variance_point=min(lambda_values)
    plt.axvline(np.log10(High_Variance_point), color='Blue', linestyle='--', label='High Variance')
    
    
    
    
    plt.xlabel('log(\lambda)')
    plt.ylabel('Average Mean Squared Error')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

lambda_values = [10**(-15), 10**(-9), 10**(-6), 10**(-3), 10**(-2), 10**(-1), 1, 10, 100, 1000, 10**6, 10**9, 10**15]
avg_training_errors = []
avg_validation_errors = []

for i in lambda_values:
    Training_scores, Validation_scores = KFold_Poly(Data, i)
    avg_training_errors.append(sum(Training_scores)/5)
    avg_validation_errors.append(sum(Validation_scores)/5)
    print(f" Lambda : {i} , Avg Training_Score : {sum(Training_scores)/5}, Avg Validation_Score : {sum(Validation_scores)/5}")
plot_learning_curve(lambda_values, avg_training_errors, avg_validation_errors)


# 4e) Plot the fitted curve to the given data ($\hat{y}$ against $x$) for the three models reported in part (e).

# In[ ]:


fig, axes = plt.subplots(1, 3, figsize=(18, 6))

def plot_curve(ax,x,y,weight,title):
    
    matrix=designMatrix(x,24)
    
    
    y_pred=np.dot(matrix,weight)
    ax.scatter(x,y_pred,label='Fitted Curve', color='red')
    ax.scatter(x,y,label='Data')
    
    
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.grid(True)

lambda_values = [10**(15), 10**(-15), 10**(-6)]
weights = []
for i in lambda_values:
    weights.append(polyregression(x,y, i, 24))
    
plot_curve(axes[0],x,y,weights[0],"High Bias Model (λ = 10**15)")
plot_curve(axes[1],x,y,weights[1],"High Variance Model (λ = 10**(-15))")
plot_curve(axes[2],x,y,weights[2],"Best Model(λ = 10**(-6))")
    


# In[ ]:




