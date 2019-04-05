import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os



class Admission_Predictor:
    def __init__(self):
        path = '/Users/pragya/PycharmProjects/AI LAB/venv/Admission_Prediction/Data/'
        os.chdir(path)
        self.data = pd.read_csv("Admission_Predict.csv")

    def model_decision(self):
        # data Pre - processing
        cols = self.data.columns
        features = cols[1:-1]
        target = cols[-1]

        X = self.data[features]
        y = self.data['Chance of Admit ']

        # train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)
        train_X = X[:int(0.75*len(X))]
        test_X = X[int(0.75*len(X)):]
        train_y = y[:int(0.75*len(X))]
        test_y = y[int(0.75*len(X)):]

        # Training
        self.model = DecisionTreeRegressor(min_samples_leaf=10,random_state=0,max_leaf_nodes=20)
        self.model.fit(train_X, train_y)
        print(self.model)

        # Visualization
        with open("classifier1.dot", "w") as f:
            f = tree.export_graphviz(self.model, feature_names=features, class_names=target, out_file=f)


        # Test data Prediction
        self.predicted = self.model.predict(test_X)
        self.predicted_full = self.model.predict(X)

        return train_X, test_X, train_y, test_y

    # sample prediction
    def predict(self,df):
        train_X, test_X, train_y, test_y = self.model_decision()
        pred = self.model.predict(df)
        return pred
        # predicted = model.predict(train_X)

    # Actual - predicted for test/train data
    def error_calc(self, test):
        mae = mean_absolute_error(test, self.predicted)
        print("Mean Absolute error: ", mae)

    def output_results(self):
        df1 = pd.DataFrame()
        df1['predictions'] = self.predicted_full
        final_df = pd.merge(left=self.data, right=df1, left_index=True, right_index=True)
        final_df['True_Decision'] = final_df['Chance of Admit '].apply(lambda x: "Yes" if x > 0.80 else "No")
        final_df['Decision'] = final_df['predictions'].apply(lambda x: "Yes" if x > 0.80 else "No")
        # print(final_df)

        final_df.to_csv('Acceptance_Prediction1.csv', index=False)


ad = Admission_Predictor()

train_X, test_X, train_y, test_y = ad.model_decision()
# 
# print(type(train_X))
# print(len(ad.predicted_full))
ad.error_calc(test_y)
ad.output_results()
# 
# # l = [310,108,4,4.5,4.5,8.61,0]
# l= [337,118,4,4.5,4.5,9.65,1]
# 
# df= pd.DataFrame(columns=['A','B','C','D','E','F','G'])
# 
# df = df.append(pd.DataFrame([l],columns=df.columns))
# print(df)
# pred = ad.predict(df)
# print("Calculated Acceptance rate: " ,",".join([str(i) for i in pred]))
# decision = "High" if pred[0] > 0.80 else "Low"
# print(decision)
