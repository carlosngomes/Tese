import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import scipy.signal as signal
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score, confusion_matrix
import pandas as pd

def getpsd(data, fs):
    """
    Obtains Power Spectral Density (PSD) using Welch method.

    Parameters:
    - data (dict): Dictionary containing data for each channel.
                   Outer dictionary keys represent names.
                   Inner dictionary keys represent channel names.
                   Inner dictionary values represent data arrays.
    - fs (float): Sampling frequency of the data.

    Returns:
    - f (dict): Dictionary containing frequencies for each name.
    - S (dict): Dictionary containing PSD for each name and channel.
    """
    # Initialize dictionaries to store frequencies and PSD values
    f = {}
    S = {}

    # Iterate over each name in the data dictionary
    for name, values_dict in data.items():
        # Initialize dictionary to store PSD values for each channel
        channel_s = {}
        # Iterate over each channel in the values dictionary
        for channel_name, channel_data in values_dict.items():
            # Compute PSD using Welch method
            frequencies, psd = signal.welch(channel_data, fs)
            # Store frequencies and PSD values in the channel dictionary
            f[name] = frequencies
            channel_s[channel_name] = psd
        # Store PSD values for each channel in the name dictionary
        S[name] = channel_s
    
    # Return dictionaries containing frequencies and PSD values
    return f, S




def getdataperchannel(S):
    """
    Extracts data per channel from the PSD data.

    Parameters:
    - S (dict): Dictionary containing PSD data.
                Outer dictionary keys represent subject IDs.
                Inner dictionary keys represent channel names.
                Inner dictionary values represent PSD arrays.
    - channel_names (list): List of channel names to extract data for.

    Returns:
    - data_per_channel (dict): Dictionary containing data per channel for each subject.
                               Outer dictionary keys represent subject IDs.
                               Inner dictionary keys represent channel names.
                               Inner dictionary values represent data arrays.
    """
    # Initialize dictionary to store data per channel
    data_per_channel = {}

    # Iterate over each subject in the PSD data dictionary
    for subject_id, subject_data in S.items():
        # Initialize dictionary to store data for each channel
        channel_data1 = {}

        # Iterate over each channel in the subject data
        for channel_name, channel_data in subject_data.items():
            # Initialize empty array to store data for the current channel
            array = []

            # Iterate over each PSD value (event) for the current channel
            for j in range(len(channel_data)):
                # Extract data for the current channel and for each event
                data = channel_data[j]
                array.append(data)

            # Convert the list of data arrays into a numpy array
            array = np.array(array)
            # Store the data array for the current channel
            channel_data1[channel_name] = array

        # Store data for each channel in the subject's data dictionary
        data_per_channel[subject_id] = channel_data1

    # Return dictionary containing data per channel for each subject
    return data_per_channel




def delta(data, freq_bands, f):
    """
    Calculate the mean power within the delta frequency band for each data point.

    Parameters:
    - data (array-like): Array containing PSD data for each data point.
    - freq_bands (dict): Dictionary containing frequency band ranges.
    - f (array-like): Array containing frequency values.

    Returns:
    - data_list (list): List containing the mean power within the delta frequency band for each data point.
    """
    data_list = []
    for i in range(len(data)):
        lower_bound = float(freq_bands['delta'][0])
        higher_bound = float(freq_bands['delta'][1])
        indices = np.where((f >= lower_bound) & (f <= higher_bound))
        y_selected = data[i][indices]
        data_list.append(np.mean(y_selected))
    return data_list

def theta(data, freq_bands,f): 
    """
    Calculate the mean power within the delta frequency band for each data point.

    Parameters:
    - data (array-like): Array containing PSD data for each data point.
    - freq_bands (dict): Dictionary containing frequency band ranges.
    - f (array-like): Array containing frequency values.

    Returns:
    - data_list (list): List containing the mean power within the delta frequency band for each data point.
    """
    data_list=[]
    for i in range(len(data)):
        lower_bound=float(freq_bands['theta'][0])
        higher_bound= float(freq_bands['theta'][1])
        indices = np.where((f >= lower_bound) & (f <= higher_bound))  
        y_selected = data[i][indices]
        data_list.append(np.mean(y_selected))
    return data_list

def alpha(data, freq_bands,f): 
    """
    Calculate the mean power within the delta frequency band for each data point.

    Parameters:
    - data (array-like): Array containing PSD data for each data point.
    - freq_bands (dict): Dictionary containing frequency band ranges.
    - f (array-like): Array containing frequency values.

    Returns:
    - data_list (list): List containing the mean power within the delta frequency band for each data point.
    """
    data_list=[]
    for i in range(len(data)):
        lower_bound=float(freq_bands['alpha'][0])
        higher_bound= float(freq_bands['alpha'][1])
        indices = np.where((f >= lower_bound) & (f <= higher_bound))   
        y_selected = data[i][indices]
        data_list.append(np.mean(y_selected))
    return data_list

def beta(data, freq_bands,f): 
    """
    Calculate the mean power within the delta frequency band for each data point.

    Parameters:
    - data (array-like): Array containing PSD data for each data point.
    - freq_bands (dict): Dictionary containing frequency band ranges.
    - f (array-like): Array containing frequency values.

    Returns:
    - data_list (list): List containing the mean power within the delta frequency band for each data point.
    """
    data_list=[]
    for i in range(len(data)):
        lower_bound=float(freq_bands['beta'][0])
        higher_bound= float(freq_bands['beta'][1])
        indices = np.where((f >= lower_bound) & (f <= higher_bound)) 
        y_selected = data[i][indices]
        data_list.append(np.mean(y_selected))  
    return data_list

def high(data, freq_bands,f): 
    """
    Calculate the mean power within the delta frequency band for each data point.

    Parameters:
    - data (array-like): Array containing PSD data for each data point.
    - freq_bands (dict): Dictionary containing frequency band ranges.
    - f (array-like): Array containing frequency values.

    Returns:
    - data_list (list): List containing the mean power within the delta frequency band for each data point.
    """
    data_list=[]
    for i in range(len(data)):
        lower_bound=float(freq_bands['high'][0])
        higher_bound= float(freq_bands['high'][1])
        indices = np.where((f >= lower_bound) & (f <= higher_bound))  
        y_selected = data[i][indices]
        data_list.append(np.mean(y_selected))  
    return data_list

def All(data, freq_bands,f): 
    """
    Calculate the mean power within the delta frequency band for each data point.

    Parameters:
    - data (array-like): Array containing PSD data for each data point.
    - freq_bands (dict): Dictionary containing frequency band ranges.
    - f (array-like): Array containing frequency values.

    Returns:
    - data_list (list): List containing the mean power within the delta frequency band for each data point.
    """
    data_list=[]
    for i in range(len(data)):
        lower_bound=float(freq_bands['all'][0])
        higher_bound= float(freq_bands['all'][1])
        indices = np.where((f >= lower_bound) & (f <= higher_bound))    
        y_selected = data[i][indices]
        data_list.append(np.mean(y_selected))
    return data_list

def low(data, freq_bands,f): 
    """
    Calculate the mean power within the delta frequency band for each data point.

    Parameters:
    - data (array-like): Array containing PSD data for each data point.
    - freq_bands (dict): Dictionary containing frequency band ranges.
    - f (array-like): Array containing frequency values.

    Returns:
    - data_list (list): List containing the mean power within the delta frequency band for each data point.
    """
    data_list=[]
    for i in range(len(data)):
        lower_bound=float(freq_bands['low'][0])
        higher_bound= float(freq_bands['low'][1])
        indices = np.where((f >= lower_bound) & (f <= higher_bound))    
        y_selected = data[i][indices]
        data_list.append(np.mean(y_selected))
    return data_list




def fcz_features(type, dict, freq_bands, freq): 
    """
    Calculate mean power for specified frequency bands for FCZ cluster or other channels.

    Parameters:
    - type (str): Type of features to compute ('theta', 'all', 'high', 'other').
    - dict (dict): Dictionary containing PSD data for each channel of each subject.
    - freq_bands (dict): Dictionary containing frequency band ranges.
    - freq (array-like): Array containing frequency values.

    Returns:
    - dados_mean (dict): Dictionary containing mean power for each subject.
    """
    dados_mean = {}  # Initialize dictionary to store mean power for each subject
    Cluster_FCZ = ['FZ', 'FC1', 'FCZ', 'FC2', 'CZ']  # Define FCZ cluster channels
    for subject_id, values in dict.items():  # Iterate over each subject in the data dictionary
        values_list = []  # Initialize list to store mean power for each channel
        for channel_name, channel_data in values.items():  # Iterate over each channel's PSD data
            if type == 'theta':
                data = theta(channel_data, freq_bands, freq)  # Calculate mean power for theta band
            elif type == 'all':
                data = All(channel_data, freq_bands, freq)  # Calculate mean power for all frequency bands
            elif type == 'high':
                data = high(channel_data, freq_bands, freq)  # Calculate mean power for high frequency band
            elif type == 'other':
                data = theta(channel_data, freq_bands, freq)  # Calculate mean power using theta function for 'other' channels
            else:
                assert False, "Invalid type '{}' provided.".format(type)  # Raise error for invalid type
            if (type == 'other' and channel_name not in Cluster_FCZ) or (channel_name in Cluster_FCZ):
                values_list.append(data)  # Append mean power to list if it's in the FCZ cluster or it's an 'other' channel
            else:
                continue  # Skip channels that are not in the FCZ cluster for 'other' type
        means = mean_of_lists(values_list)  # Calculate mean of mean power for all channels
        dados_mean[subject_id] = means  # Store mean power for each subject
    return dados_mean  # Return dictionary containing mean power for each subject

def all_features(ch_name,type, dict, freq_bands, freq): 
    """
    Calculate mean power for specified frequency bands for each channel of each subject.

    Parameters:
    - type (str): Type of features to compute ('theta', 'delta', 'all', 'high', 'alpha').
    - dict (dict): Dictionary containing PSD data for each channel of each subject.
    - freq_bands (dict): Dictionary containing frequency band ranges.
    - freq (array-like): Array containing frequency values.

    Returns:
    - dados_mean (dict): Dictionary containing mean power for each subject.
    """
    dados_mean = {}  # Initialize dictionary to store mean power for each subject
    for subject_id, values in dict.items():  # Iterate over each subject in the dictionary
        values_list = []  # Initialize list to store mean power for each channel
        for channel_name, channel_data in values.items():  # Iterate over each channel's PSD data
            if type == 'theta':
                data = theta(channel_data, freq_bands, freq)  # Calculate mean power for theta band
            elif type == 'delta':
                data = delta(channel_data, freq_bands, freq)  # Calculate mean power for delta band
            elif type == 'all':
                data = All(channel_data, freq_bands, freq)  # Calculate mean power for all frequency bands
            elif type == 'high':
                data = high(channel_data, freq_bands, freq)  # Calculate mean power for high frequency band
            elif type == 'alpha':
                data = alpha(channel_data, freq_bands, freq)  # Calculate mean power for alpha band
            else:
                assert False, "Invalid type '{}' provided.".format(type)  # Raise error for invalid type
            if channel_name==ch_name:
                values_list.append(data)  # Append mean power to list for each channel
            else:
                continue
        means = mean_of_lists(values_list)  # Calculate mean of mean power for all channels
        dados_mean[subject_id] = means  # Store mean power for each subject
    return dados_mean  # Return dictionary containing mean power for each subject

def midfrontal_features(ch_name,type, dict, freq_bands, freq): 
    """
    Calculate mean power for specified frequency bands for midfrontal cluster or other channels.

    Parameters:
    - type (str): Type of features to compute ('theta', 'other').
    - dict (dict): Dictionary containing PSD data for each channel of each subject.
    - freq_bands (dict): Dictionary containing frequency band ranges.
    - freq (array-like): Array containing frequency values.

    Returns:
    - dados_mean (dict): Dictionary containing mean power for each subject.
    """
    dados_mean = {}  # Initialize dictionary to store mean power for each subject
    Cluster_midfrontal = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F5', 'F3', 'F1', 'F2', 'FZ', 'F4', 'F6', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'C1', 'CZ', 'C2']  # Define midfrontal cluster channels
    for subject_id, values in dict.items():  # Iterate over each subject in the dictionary
        values_list = []  # Initialize list to store mean power for each channel
        for channel_name, channel_data in values.items():  # Iterate over each channel's PSD data
            if type == 'theta':
                data = theta(channel_data, freq_bands, freq)  # Calculate mean power for theta band
            elif type == 'other':
                data = theta(channel_data, freq_bands, freq)  # Calculate mean power using theta function for 'other' channels
            else:
                assert False, "Invalid type '{}' provided.".format(type)  # Raise error for invalid type
            if (type == 'other' and channel_name not in Cluster_midfrontal) or (type=='theta' and channel_name==ch_name):
                values_list.append(data)  # Append mean power to list if it's in the midfrontal cluster or it's an 'other' channel
            else:
                continue  # Skip channels that are not in the midfrontal cluster for 'other' type
        means = mean_of_lists(values_list)  # Calculate mean of mean power for all channels
        dados_mean[subject_id] = means  # Store mean power for each subject
    return dados_mean  # Return dictionary containing mean power for each subject

def low_features(ch_name,type, dict, freq_bands, freq):
    """
    Calculate mean power for specified frequency bands for low channels.

    Parameters:
    - type (str): Type of features to compute ('alpha', 'delta', 'low', 'all', 'high').
    - dict (dict): Dictionary containing PSD data for each channel of each subject.
    - freq_bands (dict): Dictionary containing frequency band ranges.
    - freq (array-like): Array containing frequency values.

    Returns:
    - dados_mean (dict): Dictionary containing mean power for each subject.
    """
    dados_mean = {}  # Initialize dictionary to store mean power for each subject
    for subject_id, values in dict.items():  # Iterate over each subject in the dictionary
        values_list = []  # Initialize list to store mean power for each channel
        for channel_name, channel_data in values.items():  # Iterate over each channel's PSD data
            if type == 'alpha':
                data = alpha(channel_data, freq_bands, freq)  # Calculate mean power for alpha band
            elif type == 'delta':
                data = delta(channel_data, freq_bands, freq)  # Calculate mean power for delta band
            elif type == 'low':
                data = low(channel_data, freq_bands, freq)  # Calculate mean power for low frequency band
            elif type == 'all':
                data = All(channel_data, freq_bands, freq)  # Calculate mean power for all frequency bands
            elif type == 'high':
                data = high(channel_data, freq_bands, freq)  # Calculate mean power for high frequency band
            else:
                assert False, "Invalid type '{}' provided.".format(type)  # Raise error for invalid type
            if channel_name==ch_name:
                values_list.append(data)  # Append mean power to list for each channel
            else:
                continue
        means = mean_of_lists(values_list)  # Calculate mean of mean power for all channels
        dados_mean[subject_id] = means  # Store mean power for each subject
    return dados_mean  # Return dictionary containing mean power for each subject




def feature(type,ch_name1,ch_name2, type_freq1, type_freq2, data1, data2, freq_bands, f1, f2): 
    """
    Calculate feature values for specified types of features and frequency bands.

    Parameters:
    - type (str): Type of feature computation ('fcz_features', 'all_features', 'midfrontal_features', 'low_features').
    - type_freq1 (str): Type of features to compute for data1 ('theta', 'delta', 'all', 'high', 'other').
    - type_freq2 (str): Type of features to compute for data2 ('theta', 'delta', 'all', 'high', 'other').
    - data1 (dict): Dictionary containing PSD data for each channel of each subject for data1.
    - data2 (dict): Dictionary containing PSD data for each channel of each subject for data2.
    - freq_bands (dict): Dictionary containing frequency band ranges.
    - f1 (dict): Dictionary containing frequency values for data1.
    - f2 (dict): Dictionary containing frequency values for data2.

    Returns:
    - feature (dict): Dictionary containing feature values for each subject.
    """
    feature = {}  # Initialize dictionary to store feature values for each subject
    for subject_id in data1.keys():  # Iterate over each subject in the data1 dictionary
        if type == 'fcz_features':
            # Calculate feature values for FCZ cluster features
            feature[subject_id] = [x/y for x, y in zip(fcz_features(type_freq1, data1, freq_bands, f1[subject_id])[subject_id], fcz_features(type_freq2, data2, freq_bands, f2[subject_id])[subject_id])]
        elif type == 'all_features':
            # Calculate feature values for all channels features
            feature[subject_id] = [x/y for x, y in zip(all_features(ch_name1,type_freq1, data1, freq_bands, f1[subject_id])[subject_id], all_features(ch_name2,type_freq2, data2, freq_bands, f2[subject_id])[subject_id])]
        elif type == 'midfrontal_features':
            # Calculate feature values for midfrontal cluster features
            feature[subject_id] = [x/y for x, y in zip(midfrontal_features(ch_name1,type_freq1, data1, freq_bands, f1[subject_id])[subject_id], midfrontal_features(ch_name2,type_freq2, data2, freq_bands, f2[subject_id])[subject_id])]
        elif type == 'low_features':
            # Calculate feature values for low channels features
            feature[subject_id] = [x/y for x, y in zip(low_features(ch_name1,type_freq1, data1, freq_bands, f1[subject_id])[subject_id], low_features(ch_name2,type_freq2, data2, freq_bands, f2[subject_id])[subject_id])]
        else:
            assert False, "Invalid type '{}' provided.".format(type)  # Raise error for invalid type
    return feature  # Return dictionary containing feature values for each subject




def save_dict_to_file(dictionary, filename):
    """
    Save dictionary to file in a specific format.

    Parameters:
    - dictionary (dict): Dictionary to be saved to file.
    - filename (str): Name of the file to save the dictionary to.
    """
    with open(filename, 'w') as f:  # Open the file in write mode
        for subject, events in dictionary.items():  # Iterate over each subject in the dictionary
            f.write(f"{subject}:")  # Write the subject ID to the file
            for event in events:  # Iterate over each event for the current subject
                if isinstance(event, list):  # Check if the event is a list
                    f.write("[" + ",".join(map(str, event)) + "]")  # Write the event list as a string
                else:
                    f.write("[" + str(event) + "]")  # Write the event as a string
            f.write("\n")  # Write a newline character after all events for the current subject

def load_dict_from_file(filename):
    """
    Load dictionary from file in a specific format.

    Parameters:
    - filename (str): Name of the file to load the dictionary from.

    Returns:
    - loaded_dict (dict): Dictionary loaded from the file.
    """
    loaded_dict = {}  # Initialize an empty dictionary to store loaded data
    with open(filename, 'r') as f:  # Open the file in read mode
        for line in f:  # Iterate over each line in the file
            parts = line.strip().split(':')  # Split each line by colon ':' to separate subject and events
            subject = parts[0].strip()  # Extract subject ID from the first part
            events = [event.strip('][').split('][') for event in parts[1:]]  # Split events by ']['
            # Split each feature and convert to float
            events = [[float(feature) for feature in event.split(',')] for sublist in events for event in sublist]
            loaded_dict[subject] = events  # Store the events for the subject in the dictionary
    return loaded_dict  # Return the loaded dictionary




def extend_list(dict):
    """
    Extend a list by appending all elements from lists stored as values in a dictionary.

    Parameters:
    - dict (dict): Dictionary containing lists as values.

    Returns:
    - all_subj (list): Extended list containing all elements from the lists in the dictionary.
    """
    all_subj = []  # Initialize an empty list to store all elements
    for subject_id in dict.keys():  # Iterate over each key (subject ID) in the dictionary
        all_subj.extend(dict[subject_id])  # Extend the list by appending all elements from the list associated with the subject ID
    return all_subj  # Return the extended list




def mean_of_lists(input_list):
    """
    Calculate the element-wise mean of multiple lists.

    Parameters:
    - input_list (list of lists): List containing multiple lists.

    Returns:
    - result (list): List containing the element-wise mean of the input lists.
    """
    # Calculate the sum of corresponding elements in each list, then divide by the number of lists
    result = [sum(items) / len(items) for items in zip(*input_list)]
    return result  # Return the element-wise mean list




def classification(i,X, Y):
    """
    Perform classification using Support Vector Machine (SVM) with different kernels.

    Parameters:
    - kern (str): Kernel type for SVM ('linear', 'rbf', etc.).
    - feature (dict): Dictionary containing feature data for each subject.
    - labels (dict): Dictionary containing labels for each subject.

    Returns:
    None
    """
    

    f1score = []
    precision = []
    recall = []
    specificity = []
    npv = []
    bal_acc = []

    splits = 100
    sss = StratifiedShuffleSplit(n_splits=splits, test_size=0.3, random_state=0)
    svm = SVC(class_weight='balanced')
    scaler = StandardScaler()
    X_col = X.columns
    Y=np.array(Y)

    for train_index, test_index in sss.split(X, Y):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        Y_train, Y_test = Y[train_index], Y[test_index]
        scal = scaler.fit(X_train)
        X_train = scal.transform(X_train)  # Variables standardization
        X_test = scal.transform(X_test)  # Variables standardization
        X_train = pd.DataFrame(X_train, columns=X_col)
        X_test = pd.DataFrame(X_test, columns=X_col)
        clf = svm.fit(X_train, Y_train)
        Y_predicted = clf.predict(X_test)
        f1score.append(f1_score(Y_test, Y_predicted))
        precision.append(precision_score(Y_test, Y_predicted))  # Precision = Positive predictive value
        npv.append(precision_score(Y_test, Y_predicted, pos_label=0))  # Negative predictive value
        recall.append(recall_score(Y_test, Y_predicted))  # Recall = Sensitivity
        specificity.append(recall_score(Y_test, Y_predicted, pos_label=0))
        bal_acc.append(balanced_accuracy_score(Y_test, Y_predicted))

    # print(np.mean(f1score))
    # print(np.mean(precision))
    print("Number of features used: "+ str(i))
    print("Mean Sensitivity: {:.4f}".format(np.mean(recall)))
    print("Std Sensitivity: {:.4f}".format(np.std(recall)))
    print("Mean Specificity: {:.4f}".format(np.mean(specificity)))
    print("Std Specificity: {:.4f}".format(np.std(specificity)))
    # print(np.mean(npv))
    print("Mean Balanced Accuracy: {:.4f}".format(np.mean(bal_acc)))
    print("Std Balanced Accuracy: {:.4f}".format(np.std(bal_acc)))
