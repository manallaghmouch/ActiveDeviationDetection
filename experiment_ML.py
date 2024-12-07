import os
import csv 
import pickle
import numpy as np
import pandas as pd
import random

from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Load the dictionaries from pickle files
with open('G:\Shared drives\PhD Manal\Projects\Git\ActiveLearningInAuditing\X_df_norm_dict100.pkl', 'rb') as f:
    df_norm_dict = pickle.load(f)
with open('G:\Shared drives\PhD Manal\Projects\Git\ActiveLearningInAuditing\FINAL_df_audit_dict.pkl', 'rb') as f:
    df_audit_dict = pickle.load(f)
with open('G:\Shared drives\PhD Manal\Projects\Git\ActiveLearningInAuditing\FINAL_deviations_dict.pkl', 'rb') as f:
    deviations_dict = pickle.load(f)
with open('G:\Shared drives\PhD Manal\Projects\Git\ActiveLearningInAuditing\FINAL_traces_dict.pkl', 'rb') as f:
    traces_dict = pickle.load(f)
with open('G:\Shared drives\PhD Manal\Projects\Git\ActiveLearningInAuditing\X_eventlog_dict100.pkl', 'rb') as f:
    eventlog_dict = pickle.load(f)

# Input parameters
num_logs = 100 # number of logs to create and to use for classification
percentages = 0.05 # percentages = [0.01, 0.02, 0.05]
total_iterations = 50 # aantal keer dat we training set trekken uit populatie obv de training set size voor een gegeven log 
deletion_percentages = [0.25,0.50,0.75]
initial_training_set_size = [50,75,100,125]

number_of_repetitions = 5 # aantal keer dat we performance schatten om daarna het gemiddelde te nemen
performance_metrics = ["accuracy", "precision", "recall", "f1_score"]

number_of_AL_iterations = 11
number_of_cases_per_iteration = 5

iter_precision_90 = -1 #start value: means that it was never reached
iter_precision_95 = -1 
iter_recall_90 = -1
iter_recall_95 = -1

# Dictionaries and lists to store results
results_RF_dict = {}
perc_anomalies_in_sample_dict = {}
len_deviations_dict = {}

# Active learning results
for h in range(1, num_logs + 1):
    if h in traces_dict.keys():
        results_RF_dict[h] = {}
        # results_AL_dict[h] = {}
        perc_anomalies_in_sample_dict[h] = {}
        len_deviations_dict[h] = {}

        for deletion_percentage in deletion_percentages:
            if deletion_percentage in traces_dict[h].keys():
                results_RF_dict[h][deletion_percentage] = {}
                # results_AL_dict[h][deletion_percentage] = {}
                perc_anomalies_in_sample_dict[h][deletion_percentage] = {}
                len_deviations_dict[h][deletion_percentage] = {}

                for i in range(1, total_iterations+1):
                    results_RF_dict[h][deletion_percentage][i] = {}

                    len_deviations = random.randint(25, 150)
                    
                    if len_deviations < len(traces_dict[h][deletion_percentage]):
                                                
                        precision_per_iteration = []
                        recall_per_iteration = []
                        
                        len_deviations_dict[h][deletion_percentage][i] = len_deviations
                        data = traces_dict[h][deletion_percentage].sample(len_deviations)

                        label_index = data.columns.get_loc('label')
                        features = data.iloc[:, label_index + 1:]
                        X_train = features
                        y_train = data['label']
                        perc_anomalies_in_sample_dict[h][deletion_percentage][i] = np.sum(y_train) / len(y_train)

                        training_examples = data['case:concept:name'].tolist()
                        population = traces_dict[h][deletion_percentage]
                        test_set = population[~population['case:concept:name'].isin(training_examples)]

                        label_index_test = test_set.columns.get_loc('label')
                        features_test = test_set.iloc[:, label_index + 1:]
                        X_test = features_test
                        y_test = test_set['label']

                        nr_of_y_values = len(np.unique(y_train))

                        if nr_of_y_values > 1:
                            X_resampled, y_resampled = RandomOverSampler().fit_resample(X_train, y_train)

                            iteration = 1  # Active learning iteration count

                            while iteration <= number_of_AL_iterations:
                                print(f"--- Iteration {iteration} ---")
                                iter_AL = iteration

                                # 1. INITIAL TRAINING
                                temp_dict = {}
                            
                                # apply classifier multiple times 
                                for j in range(1, number_of_repetitions + 1):
                                    clf = RandomForestClassifier()
                                    clf = clf.fit(X_resampled,y_resampled)
                                    y_pred = clf.predict(X_test)

                                    # print(clf.feature_importances_)

                                    nr_of_values_predicted = len(np.unique(y_pred))
                                    nr_of_values_true = len(np.unique(y_test))

                                    for k in performance_metrics: 
                                        temp_dict[k] = {}
                                        if k == "accuracy":
                                            temp_dict[k][j] = metrics.accuracy_score(y_test, y_pred)
                                        elif k == "precision":
                                            temp_dict[k][j] =  metrics.precision_score(y_test, y_pred) if nr_of_values_predicted > 1 else 0
                                        elif k == "recall":
                                            temp_dict[k][j] = metrics.recall_score(y_test, y_pred) if nr_of_values_predicted > 1 else 0
                                        elif k == "f1_score":
                                            temp_dict[k][j] = metrics.f1_score(y_test, y_pred) if nr_of_values_predicted > 1 else 0

                                # Calculate mean performance
                                metrics_dict = {}
                                for k in performance_metrics:
                                    if k == "accuracy":
                                        metrics_dict[k] = np.array(list(temp_dict["accuracy"].values())).mean()
                                    elif k == "precision":
                                        metrics_dict[k] = np.array(list(temp_dict["precision"].values())).mean()
                                    elif k == "recall":
                                        metrics_dict[k] = np.array(list(temp_dict["recall"].values())).mean()
                                    elif k == "f1_score":
                                        metrics_dict[k] = np.array(list(temp_dict["f1_score"].values())).mean()
                                

                                if iteration == 1: 
                                    initial_precision = metrics_dict["precision"]
                                    initial_recall = metrics_dict["recall"]
                                    if initial_precision < 0.6 or initial_recall < 0.6:
                                        metrics_dict["performance_lower_60"] = 1
                                        precision_per_iteration.append(metrics_dict["precision"])
                                        recall_per_iteration.append(metrics_dict["recall"])
                                        break
                                    else: 
                                        metrics_dict["performance_lower_60"] = 0

                                print("Precision: " + str(metrics_dict["precision"]))
                                print("Recall: " + str(metrics_dict["recall"]))

                                precision_per_iteration.append(metrics_dict["precision"])
                                recall_per_iteration.append(metrics_dict["recall"])
                                
                                if metrics_dict["precision"] >= 0.9:
                                    iter_precision_90 = iteration
                                    if metrics_dict["precision"] >= 0.95:
                                        iter_precision_95 = iteration

                                if metrics_dict["recall"] >= 0.9:
                                    iter_recall_90 = iteration
                                    if metrics_dict["recall"] >= 0.95:
                                        iter_recall_95 = iteration

                                # Stop if precision and recall are both 1
                                if metrics_dict["precision"] == 1.0 and metrics_dict["recall"] == 1:
                                    iteration += 1
                                    print("Stopping early as precision and recall are both 1.")
                                    break

                                # 2. Add randomnly cases to training set (aangepast)
                                available_test_rows = len(test_set)
                                
                                if available_test_rows >= number_of_cases_per_iteration:
                                    new_training_data = test_set.sample(n=number_of_cases_per_iteration, replace=False)
                                    print(f"Adding {number_of_cases_per_iteration} cases to the training set.")
                                else:
                                    print(f"Test set too small to proceed.")
                                    population_empty = 1 
                                    break

                                new_training_data = test_set.sample(n=number_of_cases_per_iteration)
                                X_new_train = new_training_data.iloc[:, label_index + 1:]
                                y_new_train = new_training_data['label']

                                # Update training set with least confident examples
                                X_train = pd.concat([X_train, X_new_train])
                                y_train = pd.concat([y_train, y_new_train])

                                # Remove added samples from test set
                                test_set = test_set[~test_set['case:concept:name'].isin(new_training_data['case:concept:name'])]
                                X_test = test_set.iloc[:, label_index + 1:]
                                y_test = test_set['label']

                                # Stop if test set is empty
                                if test_set.empty:
                                    iteration += 1
                                    print("Test set is empty; stopping active learning.")
                                    break

                                # Increment the iteration count
                                iteration += 1

                                results_RF_dict[h][deletion_percentage][i][iter_AL] = metrics_dict # last saved metrics (final ones)

for h in range(1, num_logs + 1):
    if h in results_RF_dict:
        for value in deletion_percentages:
            if value in results_RF_dict[h]:
                for i in range(1, total_iterations + 1):
                    if i in results_RF_dict[h][value]:
                        file_path = f"test_result_{h}_{value}_{i}_{iter_AL}.csv"
                        
                        # Initialize the CSV file with headers
                        if not os.path.exists(file_path):
                            with open(file_path, 'w', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow([
                                    "log_id", "training_set", "iteration_AL", "deletion_percentage",
                                    "num_traces_log", "num_events_log", "num_activities_model",
                                    "num_constraints_norm", "num_constraints_audit", "labeled_population",
                                    "training_set_size", "perc_anomalies_in_population", "perc_anomalies_in_sample",
                                    "accuracy_RF", "precision_RF", "recall_RF", "F1_score_RF",
                                    "number_of_AL_iter", "cases_per_iter", "total_number_labeled_cases"
                                ])

                        for iter_AL in range(1, number_of_AL_iterations + 1):
                            if iter_AL in results_RF_dict[h][value][i]:
                                # Gather data
                                log_id = h
                                training_set = i
                                iteration_AL = iter_AL
                                deletion_percentage = value
                                num_traces_log = 10000
                                num_events_log = len(eventlog_dict[h])
                                num_activities_model = len(eventlog_dict[h]['concept:name'].unique())
                                num_constraints_norm = len(df_norm_dict[h].columns)
                                num_constraints_audit = len(df_audit_dict[h][value].columns)
                                labeled_population = len(traces_dict[h][value])
                                labeled_sample = len_deviations_dict[h][value][i]
                                perc_anomalies_in_population = 0.05
                                perc_anomalies_in_sample = perc_anomalies_in_sample_dict[h][value][i]

                                accuracy_RF = results_RF_dict[h][value][i][iter_AL]["accuracy"]
                                precision_RF = results_RF_dict[h][value][i][iter_AL]["precision"]
                                recall_RF = results_RF_dict[h][value][i][iter_AL]["recall"]
                                f1_score_RF = results_RF_dict[h][value][i][iter_AL]["f1_score"]

                                number_AL_iter = number_of_AL_iterations
                                cases_per_iter = number_of_cases_per_iteration
                                total_number_labeled_cases = labeled_sample + (iter_AL * cases_per_iter)

                                # Write data to CSV
                                with open(file_path, 'a', newline='') as f:
                                    writer = csv.writer(f)
                                    writer.writerow([
                                        log_id, training_set, iteration_AL, deletion_percentage,
                                        num_traces_log, num_events_log, num_activities_model,
                                        num_constraints_norm, num_constraints_audit, labeled_population,
                                        labeled_sample, perc_anomalies_in_population, perc_anomalies_in_sample,
                                        accuracy_RF, precision_RF, recall_RF, f1_score_RF,
                                        number_AL_iter, cases_per_iter, total_number_labeled_cases
                                    ])

                                print("writing data...")

# Create dataframe from stored csv files 
def combine_result_files(directory):
    df=pd.DataFrame()
    os.chdir(directory)
    for file in os.listdir():
        if file.startswith('test_result') & file.endswith('csv'):
            aux=pd.read_csv(file)
            aux.reset_index(drop=True,inplace=True)
            df=pd.concat([df,aux])
            print("added line")
    return df

directory = 'G:\\Shared drives\\PhD Manal\\Projects\\Git\\ActiveLearningInAuditing'

combined_df = combine_result_files(directory)

columns = {
    "log_id": [],
    "training_set": [],
    "iteration_AL": [],
    "deletion_percentage": [],
    "num_traces_log": [],
    "num_events_log": [],
    "num_activities_model": [],
    "num_constraints_norm": [],
    "num_constraints_audit": [],
    "labeled_population":[],
    "training_set_size":[],
    "perc_anomalies_in_population":[],
    "perc_anomalies_in_sample":[],
    "accuracy_RF": [],
    "precision_RF": [],
    "recall_RF": [],
    "F1_score_RF": [],
    "number_of_AL_iter": [],
    "cases_per_iter": [],
    "total_number_labeled_cases": [],
}

combined_df.columns = columns
combined_df.to_csv('results_ML.csv', index=False)
