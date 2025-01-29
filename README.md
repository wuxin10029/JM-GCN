# JM-GCN
A tutorial for implementing the JMGCN model

![image](https://github.com/user-attachments/assets/ce89ec07-9945-4380-b00f-0029632ef977)


## 1. Code files
Here, we introduce the main python files for the proposed JM-GCN model.

- **data**: The data storage files.
  - `beijing_data`: Stores the original CSV files including AQI, pollutant data, and meteorological data.
  - `matrix`: Stores the three static adjacency matrices, including the distance, sequence similarity, and POI matrices.
  - `temporal_data`: Stores the sliced training, validation, and test sets.
  
- **datasets**: Houses the code for preprocessing the data, generating static matrices, and partitioning the training set.
  
- **lib**: 
  - `air.py`: For slicing the datasets.
  - `data_preprocessing.py` and `filled_data.py`: For preprocessing the data.
  - `matrix.py`: For generating the static adjacency matrix we need.
  
- **models**: Contains our proposed `JM-GCN`, as well as `ASTGCN` and `STGCN` from previous work.
  
- **wandb**: A file that holds the training losses and cases. Set `os.environ['WANDB_MODE'] = 'online'` to view training online.
  
- **generate_training_data.py**: Run to generate training data from raw data.
  
- **train.py**: The main program to start the experiment.

## 2. Dataset
We provide the raw data used by the JM-GCN model so that subsequent researchers can use the study. This includes daily pollutant and meteorological data for Beijing, site information, and POI data.
The raw data is available for download on this cloud drive 
https://drive.google.com/drive/folders/1p0a8FlJ4KN-bdqBN3BAP6YJx180vvESe?usp=sharing

Details of each file are shown below:
- **beijing_data**: You need to place `station_aq.xlsx` in this folder, get the name of the station and the latitude and longitude of the station. You need to generate a CSV file based on the raw data file with dimensions 26304*238, where 26304 is the time step and 238 means 34 stations with 7 features (including AQI, wind speed, barometric humidity, etc.) per station.
- **matrix**: `dist.npy`, `func.npy`, `poi.npy`, static adjacency matrix with dimension 34*34. Dynamic adjacency matrices are generated directly in the code run.
- **temporal_data**: Run `generate_training_data.py` to generate the training file automatically.

## 3. Running
Running `train.py` will automatically train and test the model. The result will show the MAE, MAPE, and RMSE for each step.

## 4. Related materials
(a) Runtime environment:
- python==3.7.16
- torch==1.8.1+cu111
- matplotlib==3.5.3
- keras==2.11.8
- numpy==1.21.1
- pandas==1.3.5

(b) Graphs for the experimental part:
Exported data in model training, plotted icons using python and origin.



