## why use this? 

1. Devices in the FL-based system share a major task among themselves, as opposed to the conventional approach. That is, each local device in the FL system develops a specific element of a big- ger model, which are then integrated at the FL server. Thus, the FL strategy eliminates the time and resource loss that happens when all work is done at a single location.

2. Another important point is that centralized data processing methods include limitations such as the requirement for high bandwidth, time loss, and risks to data privacy.  For example, the scalability of FL in response to an increase in the number of users has been analyzed along with its data privacy advantages. This study aims to fill gaps in the literature by focusing on the technical details of local FL simulations that are lacking.

3. users can detect threats on their own devices and contribute to the development of a larger model. Users can create their own models and send the model parameters to a central FL server. The FL server can then process local models containing information about threats from different user devices and develop a global model that can predict and block most of the threats that may occur in this area. The parameters of the global model can be sent back to all local devices in the FL system. Thus, each local device user gains the experience (without seeing the data) of all other users.


Target : ’reaching a target accuracy value’ was used as the criteria for stopping the federative system. 

## Types of classification: 
- malware detection (binary classification— BC) 
- malware type detection (multiclass classification— MC)

## 1st approach: without FL with dataset CIC-MalMem-2022

- datasets: BODMAS, Kaggle, and CIC-MalMem-2022.
1. algorithms: CatBoost (CB), XGBoost (XGB), LightGBM (LGBM), random for- est (RF), and gradient boosting model (GBM)

2. hybrid model: ML + DL 
- data balancing by oversampling (SMOTE) and dominant feature extraction with XGB have been achieved( ???? )

3. supervised and unsupervised learning algorithms
    datasets: Malware-Exploratory [12] and CIC-MalMem- 2022
    algorithms: K-means, density-based spatial clustering of applications with noise (DBSCAN), and Gaussian mixture model (GMM), and a total of seven classification algo- rithms including decision tree (DT), KNeighbors (kNN), RF, stochastic gradient descent (SGD), Ada Boost (AB), Extra Trees (ET), Gaussian Naive Bayes (GNB).

4. detect mal- ware hidden in memory (???? )
   - traditional machine learning methods were first tested, and then an extended convolutional neural network model was proposed.

5. have conducted a malware detection study based on the use of memory data (???)
    In the study where various machine learning and deep learning approaches were tried, big data platforms were also utilized
    Dataset: CIC-MalMem- 2022 dataset was used on the Google Colab platform with the PySpark big data management tool. Binary classification was performed using algorithms such as RF, DT, Gradient Boosted Tree (GBT), logistic regression (LR), Naive Bayes (NB), linear vector support machine (LVSM), multilayer per- ceptron (MLP), deep feed forward neural network (DFFNN), and long short-term memory (LSTM)

6. detect malicious software hidden in cloud environments.
    dataset: CIC-MalMem-2022
    algorithms: algorithms such as binary bat algorithm (BBA), cuckoo search algorithm (CSA), mayfly algorithm (MA), and particle swarm optimization (PSO)
    After feature selection, benign–malignant records were classified with machine learning algorithms kNN, RF, and support vector machine (SVM).

7. developed a visual-based system in their study using many datasets.
    The study proposes a com- pletely platform-independent scheme. This scheme involves extracting a file from the dumps of the operation memory and converting it to an image.
    datasets: Dumpware10, CIC-MalMem-2022, and real-world dataset of Android applications

## 2nd approach: FL without CIC-MalMem-2022
1. malware classifica- tion study
    dataset:  provided by VirusTotal, containing 10,907 records
    models: SVM and LSTM machine learning models

2. IoT malware detection study for supervised and unsupervised learning models (***)
    demon- strated that the federated learning approach maintains data privacy without compromising the model’s performance.
    dataset: N-BaIoT dataset

3. malware detection
    dataset (multiclass classification): 87,182 APK files collected from the Opera Mobile Store.
    catagories: Entertainment, Health, Games, Travel, and E-book
    approach: federated learning approach with SVM involving five users and the classic approach where data are centralized were compared.

4. less is more (LiM) malicious software classification framework (better results)
    200 users with 25K clean applications and 25K malicious applications participated in the test.
    dataset: MaMaDroid dataset

5. new classification scheme called FedHGCDroid
    classifying malicious software on Android devices based on federated learning. Initially for malicious software classification, they used HGCDroid, which extracts the behavior features of malware software, using CNN and graphic neural network (GNN). In tests with the Androzoo dataset
    dataset: Androzoo dataset

6. 

## Dataset
It represents an accurate representation of what a user would run on a computer environment during a malware attack. In the dataset of 58,296 records, there are 29,298 records that are balanced between benign and malicious (malware).


