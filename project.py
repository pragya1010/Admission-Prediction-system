import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree, linear_model
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import seaborn as sns
sns.set(style='white')
sns.set(style='whitegrid', color_codes=True)
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import linregress


class Admission_Predictor:
    def __init__(self):
        path = '/Users/pragya/PycharmProjects/AI LAB/venv/Admission_Prediction/Data/'
        os.chdir(path)
        self.data = pd.read_csv("Admission_Predict.csv")
        describe = self.data.describe().transpose()
        print(describe)

        ### Check Missing values
        print((self.data.notnull().sum() / len(self.data)).sort_values(ascending=False))

    def plot_data(self, test_y):
        cols = self.data.columns
        # print(cols)
        features = cols[1:-1]
        target = cols[-1]
        # print("Features: ", features)

        #Plots
        plt.figure(figsize=(20, 20))
        for i in range(len(features)):
            plt.subplot(3, 3, i + 1)
            # print(features[i])
            plt.scatter(self.data[features[i]], self.data['Chance of Admit '])
            plt.title(features[i])

        plt.savefig('features.pdf')
        plt.show()

        # Median student chances
        features2 = cols[[3, 4, 5, 7]]
        print('features2',features2)
        means = self.data['Chance of Admit '].mean()
        median = self.data['Chance of Admit '].median()
        print("Mean student chances", means)
        print("Median student chances",median)

        # Considering the best correalations features
        main_features = ['CGPA', 'GRE Score', 'TOEFL Score']
        for i in range(len(main_features)):
            print(main_features[i].upper())
            print(linregress(self.data[main_features[i]], self.data['Chance of Admit ']))

            plt.figure(figsize=(20, 6))
            plt.subplot(1, 2, 1)
            sns.distplot(self.data[main_features[i]], kde = False)
            plt.title('Distributed ' + main_features[i] + ' of Applicants')

            plt.subplot(1, 2, 2)
            sns.regplot(self.data[main_features[i]], self.data['Chance of Admit '])
            plt.title(main_features[i] + ' vs Chance of Admit')
            plt.savefig(main_features[i] +'.pdf')

        # Bar Plots
        df = self.data
        plt.figure(figsize=(20, 10))
        for j in range(len(features2)):
            plt.subplot(2, 2, j + 1)
            values = df[features2[j]].unique()
            ser = pd.Series(range(len(values)), index=values, dtype='float64')
            for i in range(len(values)):
                ser[values[i]] = df[df[features2[j]] == values[i]]['Chance of Admit '].mean()
            ser = ser.sort_index()

            plt.bar(ser.index, ser.values, width=0.3)
            plt.title(features2[j])
            plt.plot([0, len(values)], [median, median], 'k-', lw=1, dashes=[2, 2])

        plt.savefig('featuresVsMedian.pdf')
        plt.show()

        # Algo comparision plots
        Methods = ['Decision Tree Regression', 'Linear Regression']
        Scores = np.array([self.score1, self.score2])

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(Methods, Scores)
        plt.title('Algorithm Prediction Accuracies')
        plt.ylabel('Accuracy')
        plt.savefig("Algorithm_Prediction_Accuracies.pdf")
        plt.show()

        #Residual Plot
        plt.scatter(self.predicted_2, self.predicted_2 - test_y, c='g')
        plt.hlines(y = 0, xmin=0.4, xmax=1)
        plt.title('Residual plot')
        plt.ylabel('Residual')
        plt.savefig("Residual_LR.pdf")
        plt.show()


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


        # Decision tree Regression
        self.model = DecisionTreeRegressor(max_leaf_nodes=10)
        self.model.fit(train_X, train_y)


        #Linear Model
        self.model2 = linear_model.LinearRegression()
        self.model2.fit(train_X, train_y)


        print(self.model)
        print(self.model2)
        # print(self.model3)

        # Visualization
        with open("classifier.dot", "w") as f:
            f = tree.export_graphviz(self.model, feature_names=features, class_names=target, out_file=f)


        # Test data Prediction
        self.predicted = self.model.predict(test_X)
        self.predicted_full = self.model.predict(X)
        self.score1 = self.model.score(test_X,test_y)
        print("Score1: ", self.score1)

        self.predicted_2 = self.model2.predict(test_X)
        self.predicted_full_2 = self.model2.predict(X)
        self.score2 = self.model2.score(test_X, test_y)
        print("Score2: ", self.score2)
        print("Coefficients-----------", self.model2.coef_)
        print("Intercept-----------", self.model2.intercept_)


        return train_X, test_X, train_y, test_y


    # Actual - predicted for test/train data
    def error_calc(self, test):
        mae_DR = mean_absolute_error(test, self.predicted)
        mse_DR = mean_squared_error(test, self.predicted)
        r2_DR  = r2_score(test, self.predicted)

        mae_LR = mean_absolute_error(test, self.predicted_2)
        mse_LR = mean_squared_error(test, self.predicted_2)
        r2_LR = r2_score(test, self.predicted_2)


        print("Errors - Linear Regression: \n Mean Absolute Error: {} \n  Mean Squared Error: {} \n R2 Score: {}".format(mae_LR, mse_LR, r2_LR))
        print("Errors - Decision Tree Regression: \n Mean Absolute Error: {} \n  Mean Squared Error: {} \n R2 Score: {}".format(mae_DR, mse_DR, r2_DR))
        

    def output_results(self):
        df1 = pd.DataFrame()
        df1['predictions_DR'] = self.predicted_full
        df1['predictions_LR'] = self.predicted_full_2
        final_df = pd.merge(left=self.data, right=df1, left_index=True, right_index=True)
        final_df['True_Decision'] = final_df['Chance of Admit '].apply(lambda x: "Yes" if x > 0.80 else "No")
        final_df['Decision_DR'] = final_df['predictions_DR'].apply(lambda x: "Yes" if x > 0.80 else "No")
        final_df['Decision_LR'] = final_df['predictions_LR'].apply(lambda x: "Yes" if x > 0.80 else "No")
        print(final_df[['GRE Score','TOEFL Score','University Rating','SOP','LOR ','CGPA','Research','Chance of Admit ','predictions_LR']].tail(5))
        print(final_df[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research',
                        'Chance of Admit ', 'predictions_DR']].tail(5))

        # final_df.to_csv('Acceptance_Prediction1.csv', index=False)
        # plt.plot(final_df.index, final_df['predictions_LR'], color='red', label='Predicted')
        # plt.plot(final_df.index, final_df['Chance of Admit '], color='blue', label='Actual')
        # plt.legend()
        # plt.show()


ad = Admission_Predictor()

train_X, test_X, train_y, test_y = ad.model_decision()
# 
# print(type(train_X))
# print(len(ad.predicted_full))

ad.error_calc(test_y)
ad.output_results()
ad.plot_data(test_y)

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
