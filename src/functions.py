import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import scipy.signal as signal


def getpsd(data,fs): #Obtains PSD
    f={}
    S={}
    for name, values in data.items():
        channel_s={}
        for channel_name,values in values.items(): 
            f[name],channel_s[channel_name]= signal.welch(values,fs)
        S[name]= channel_s
    return f,S


def getdataperchannel(S,channel_names): 
    data_per_channel = {}
    for subject_id, dados in S.items():
        channel_data={}
        for channel_name, values in dados.items(): 
            array = []
            for j in range(len(values)):
                data = values[j]  # Extract data for the current channel and for each event
                array.append(data)
            array=np.array(array)
            channel_data[channel_name] = array  
        data_per_channel[subject_id] = channel_data 
    return data_per_channel  


def delta(data, freq_bands,f): 
    data_list=[]
    for i in range(len(data)):
        lower_bound=float(freq_bands['delta'][0])
        higher_bound= float(freq_bands['delta'][1])
        indices = np.where((f >= lower_bound) & (f <= higher_bound))   
        y_selected = data[i][indices]
        data_list.append(np.mean(y_selected)) 
    return data_list


def theta(data, freq_bands,f): 
    data_list=[]
    for i in range(len(data)):
        lower_bound=float(freq_bands['theta'][0])
        higher_bound= float(freq_bands['theta'][1])
        indices = np.where((f >= lower_bound) & (f <= higher_bound))  
        y_selected = data[i][indices]
        data_list.append(np.mean(y_selected))
    return data_list


def alpha(data, freq_bands,f): 
    data_list=[]
    for i in range(len(data)):
        lower_bound=float(freq_bands['alpha'][0])
        higher_bound= float(freq_bands['alpha'][1])
        indices = np.where((f >= lower_bound) & (f <= higher_bound))   
        y_selected = data[i][indices]
        data_list.append(np.mean(y_selected))
    return data_list


def beta(data, freq_bands,f): 
    data_list=[]
    for i in range(len(data)):
        lower_bound=float(freq_bands['beta'][0])
        higher_bound= float(freq_bands['beta'][1])
        indices = np.where((f >= lower_bound) & (f <= higher_bound)) 
        y_selected = data[i][indices]
        data_list.append(np.mean(y_selected))  
    return data_list


def high(data, freq_bands,f): 
    data_list=[]
    for i in range(len(data)):
        lower_bound=float(freq_bands['high'][0])
        higher_bound= float(freq_bands['high'][1])
        indices = np.where((f >= lower_bound) & (f <= higher_bound))  
        y_selected = data[i][indices]
        data_list.append(np.mean(y_selected))  
    return data_list


def All(data, freq_bands,f): 
    data_list=[]
    for i in range(len(data)):
        lower_bound=float(freq_bands['all'][0])
        higher_bound= float(freq_bands['all'][1])
        indices = np.where((f >= lower_bound) & (f <= higher_bound))    
        y_selected = data[i][indices]
        data_list.append(np.mean(y_selected))
    return data_list


def low(data, freq_bands,f): 
    data_list=[]
    for i in range(len(data)):
        lower_bound=float(freq_bands['low'][0])
        higher_bound= float(freq_bands['low'][1])
        indices = np.where((f >= lower_bound) & (f <= higher_bound))    
        y_selected = data[i][indices]
        data_list.append(np.mean(y_selected))
    return data_list


def fcz_features(tipo, data, freq_bands, freq): 
    dados_mean = {}
    Cluster_FCZ=['FZ','FC1','FCZ','FC2','CZ']
    for subject_id, values in data.items():
        dados_list=[]
        for channel_name, data in values.items():  
            if tipo == 'theta':
                dados = theta(data, freq_bands, freq)
            elif tipo == 'all':
                dados = All(data, freq_bands, freq)
            elif tipo == 'high':
                dados = high(data, freq_bands, freq)
            elif tipo == 'other':
                dados = theta(data, freq_bands, freq)  # Using theta function for 'other' channels
            else:
                assert False, "Invalid type '{}' provided.".format(tipo)     
            if (tipo == 'other' and channel_name not in Cluster_FCZ) or (channel_name in Cluster_FCZ):
                dados_list.append(dados)
            else:
                continue 
        means = mean_of_lists(dados_list)
        dados_mean[subject_id]= means
        
    return dados_mean


def all_features(tipo,data,freq_bands,freq): 
    dados_mean = {}
    for subject_id, values in data.items():
        dados_list=[]
        for channel_name, data in values.items():  
            if tipo == 'theta':
                dados = theta(data, freq_bands, freq)
            elif tipo == 'delta':
                dados= delta(data,freq_bands,freq)
            elif tipo== 'all':
                dados= All(data,freq_bands,freq)
            elif tipo=='high':
                dados= high(data,freq_bands,freq)
            elif tipo=='alpha':
                dados=alpha(data,freq_bands,freq)
            else:
                assert False, "Invalid type '{}' provided.".format(tipo)     
            dados_list.append(dados)
        means = mean_of_lists(dados_list)
        dados_mean[subject_id]= means
        
    return dados_mean


def midfrontal_features(tipo,data,freq_bands,freq): 
    dados_mean = {}
    Cluster_midfrontal=['FP1','FPZ','FP2','AF3','AF4','F5','F3','F1','F2','FZ','F4','F6','FC3','FC1','FCZ','FC2','FC4','C1','CZ','C2']
    for subject_id, values in data.items():
        dados_list=[]
        for channel_name, data in values.items():  
            if tipo == 'theta':
                dados = theta(data, freq_bands, freq)
            elif tipo == 'other':
                dados = theta(data, freq_bands, freq)  # Using theta function for 'other' 
            else:
                assert False, "Invalid type '{}' provided.".format(tipo)     
            if (tipo == 'other' and channel_name not in Cluster_midfrontal) or (channel_name in Cluster_midfrontal):
                dados_list.append(dados)
            else:
                continue 
        means = mean_of_lists(dados_list)
        dados_mean[subject_id]= means
        
    return dados_mean


def low_features(tipo,data,freq_bands,freq):
    dados_mean={}
    for subject_id, values in data.items():
        dados_list=[]
        for channel_name, data in values.items():  
            if tipo == 'alpha':
                dados = alpha(data, freq_bands, freq)
            elif tipo == 'delta':
                dados = delta(data, freq_bands, freq)
            elif tipo== 'low':
                dados = low(data,freq_bands,freq) 
            elif tipo== 'all':
                dados= All(data,freq_bands,freq)
            elif tipo== 'high':
                dados= high(data,freq_bands,freq)
            else:
                assert False, "Invalid type '{}' provided.".format(tipo)     
            
            dados_list.append(dados)
        means = mean_of_lists(dados_list)
        dados_mean[subject_id]= means
        
    return dados_mean


def feature(tipo, tipo_freq1, tipo_freq2 , data1 ,data2, freq_bands,f1,f2): 
    feature={}
    for subject_id in data1.keys():
        if tipo =='fcz_features':
            feature[subject_id]=[x/y for x,y in zip(fcz_features(tipo_freq1, data1, freq_bands, f1[subject_id])[subject_id],fcz_features(tipo_freq2, data2, freq_bands, f2[subject_id])[subject_id])]
        elif tipo=='all_features':
            feature[subject_id]=[x/y for x,y in zip(all_features(tipo_freq1, data1, freq_bands, f1[subject_id])[subject_id],all_features(tipo_freq2, data2, freq_bands, f2[subject_id])[subject_id])]

        elif tipo == 'midfrontal_features':
            feature[subject_id]=[x/y for x,y in zip(midfrontal_features(tipo_freq1, data1, freq_bands, f1[subject_id])[subject_id],midfrontal_features(tipo_freq2, data2, freq_bands, f2[subject_id])[subject_id])]
            
        elif tipo == 'low_features':
            feature[subject_id]=[x/y for x,y in zip(low_features(tipo_freq1, data1, freq_bands, f1[subject_id])[subject_id],low_features(tipo_freq2, data2, freq_bands, f2[subject_id])[subject_id])]
            
        else:
            assert False, "Invalid type '{}' provided.".format(tipo)
    return feature


def save_dict_to_file(dictionary, filename):
    with open(filename, 'w') as f:
        for subject, events in dictionary.items():
            f.write(f"{subject}:")
            for event in events:
                if isinstance(event, list):  
                    f.write("[" + ",".join(map(str, event)) + "]")
                else:
                    f.write("[" + str(event) + "]")
            f.write("\n")


def load_dict_from_file(filename):
    loaded_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(':')
            subject = parts[0].strip()
            events = [event.strip('][').split('][') for event in parts[1:]]
            # Split each feature and convert to float
            events = [[float(feature) for feature in event.split(',')] for sublist in events for event in sublist]
            loaded_dict[subject] = events
    return loaded_dict


def extend_list(dict):
    all_subj=[]
    for subject_id in dict.keys():
        all_subj.extend(dict[subject_id])
    return all_subj


def mean_of_lists(input_list):
    return [sum(items) / len(items) for items in zip(*input_list)]


def classification(kern,feature, labels):
    num_iterations = 100
    test_size = 0.3  
    train_size= 0.7

    model = SVC(kernel=kern, class_weight='balanced') #weighted SVM with rbf kernel
    scaler = StandardScaler()
    
    mean_scores = []
    std_scores = []
    sensitivities = []
    specificities = []
    balanced_accuracies = []
    coefs=[]
    value= np.array(extend_list(feature))
    label= np.array(extend_list(labels))
    
    for _ in range(num_iterations):
        shuffle_split = StratifiedShuffleSplit(test_size=test_size, train_size=train_size) #Monte-Carlo 100 iterations, 70% train, 30% test
        
        iteration_scores = []

        for train_index, test_index in shuffle_split.split(value,label):
            X_train, X_test = value[train_index], value[test_index]
            y_train, y_test = label[train_index], label[test_index]
            
            X_train_scaled = scaler.fit_transform(X_train) #Feature standardization
            X_test_scaled = scaler.transform(X_test)
            
            model.fit(X_train_scaled, y_train)
            
            score = model.score(X_test_scaled, y_test)
            iteration_scores.append(score)
            
            # Calculate confusion matrix
            y_pred = model.predict(X_test_scaled)
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            sensitivity = tp / (tp + fn)
            
            specificity = tn / (tn + fp)
            
            balanced_accuracy = (sensitivity + specificity) / 2
            
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            balanced_accuracies.append(balanced_accuracy)
            if kern=='linear':
                coef = model.coef_[0]
                coefs.append(coef)
            else: 
                continue


        mean_score = np.mean(iteration_scores)
        std_score = np.std(iteration_scores)

        print("Mean score: {:.4f}".format(mean_score))
        print("Std score: {:.4f}".format(std_score))
        print("-------------------------------")

        mean_scores.append(mean_score)
        std_scores.append(std_score)

    overall_mean_score = np.mean(mean_scores)
    overall_std_score = np.mean(std_scores)
    print("Overall Mean score: {:.4f}".format(overall_mean_score))
    print("Overall Std score: {:.4f}".format(overall_std_score))

    mean_sensitivity = np.mean(sensitivities)
    std_sensitivity = np.std(sensitivities)
    mean_specificity = np.mean(specificities)
    std_specificity = np.std(specificities)
    mean_balanced_accuracy = np.mean(balanced_accuracies)
    std_balanced_accuracy = np.std(balanced_accuracies)
    mean_coef= mean_of_lists(coefs)
    print("Mean Sensitivity: {:.4f}".format(mean_sensitivity))
    print("Std Sensitivity: {:.4f}".format(std_sensitivity))
    print("Mean Specificity: {:.4f}".format(mean_specificity))
    print("Std Specificity: {:.4f}".format(std_specificity))
    print("Mean Balanced Accuracy: {:.4f}".format(mean_balanced_accuracy))
    print("Std Balanced Accuracy: {:.4f}".format(std_balanced_accuracy))  
    if kern=='linear':
        for feature, score in enumerate(mean_coef):
            print(f"Feature {feature+1}: {score}")