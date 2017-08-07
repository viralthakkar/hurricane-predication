import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import itertools
import matplotlib.pyplot as plt

def readData(filename):
    # read the entire file into a python array
    with open(filename, 'rb') as f:
        data = f.readlines()

    # remove the trailing "\n" from each linesed '$ s/,$//g' input_file
    data = map(lambda x: x.rstrip(), data)
    return data

# atlantic File
atlantic = readData('hurdat.txt')

# pacific File
pacific = readData('hurdat2.txt')



# Extract Data for First type of line
def get_num_of_rows_follow(single_row):
    hurricane = single_row.split(",")
    single_entry = [hurricane[0][0:2],hurricane[0][2:4],hurricane[0][4:8],hurricane[1].strip(),int(hurricane[2])]
    report_hurricane_wise.append(single_entry)
    return int(hurricane[2])


# Getting only First type of line from text file
def generateReport(dataName):
    count = 0
    while(count!=len(dataName)):
        num = get_num_of_rows_follow(dataName[count])
        count = count + num + 1
    atlantic_report = report_hurricane_wise
    return atlantic_report

# Generating report of each type cyclone and number locations affected
def generateFrame(dataName):
    report_hurricane_wise = []
    report_hurricane_wise = pd.DataFrame(dataName)
    report_hurricane_wise.columns = ['Ocean Name', 'Cyclone Num', 'Year', 'Name', 'Count']
    return report_hurricane_wise

report_hurricane_wise = []
atlantic_report = generateFrame(generateReport(atlantic))
report_hurricane_wise = []
pacific_report = generateFrame(generateReport(pacific))


# Getting number of rows to follow from each first type of line
def get_rows_follow(single_row):
    hurricane = single_row.split(",")
    return int(hurricane[2])

# converting get_rows_follow to columns data
def convert_rows_columns(convert):
    analysis_data = []
    for i in range(len(convert)):
        row_data = convert[i].replace("N","")
        row_data = row_data.replace("W","")
        row_data = row_data.split(",")
        analysis_data.append(row_data)
    return analysis_data

def lat_array(lat_data):
    temp = lat_data.as_matrix()
    lat_data_array.append(temp)

def long_array(long_data):
    temp = long_data.as_matrix()
    long_data_array.append(temp)
    
def generateArray(dataName):
    count = 0
    while(count!=len(dataName)):
        num = get_rows_follow(dataName[count])
        analysis = convert_rows_columns(dataName[count+1:count+num+1])
        count = count + num + 1
        a = pd.DataFrame(analysis)
        lat_array(a[4])
        long_array(a[5])
        
def convertToDataFrame(dataName):
    data_array = []
    data_array = pd.DataFrame(dataName)
    return data_array


lat_data_array = [] 
long_data_array = []
generateArray(atlantic)
atlantic_latt = convertToDataFrame(lat_data_array)
atlantic_long = convertToDataFrame(long_data_array)

lat_data_array = [] 
long_data_array = []
generateArray(pacific)
pacific_latt = convertToDataFrame(lat_data_array)
pacific_long = convertToDataFrame(long_data_array)


# Linear Regression to predict path 
def model_prep(dataName,prediction_value,start,end):
    X = dataName[start].reshape(-1, 1)
    Y = dataName[end]
    lm = LinearRegression()
    lm.fit(X,Y)
    prediction_value = lm.predict(prediction_value)
    return prediction_value



# generating initial dataset from user input. scaling factor is 2. Thats means will consider 2 upper and lower limit to find data based on input.
def generate_new_dataset(lat,lag,region,index=0):
    new_list = []
    lat_start = int(lat) - 2
    lat_end = int(lat) + 2
    lag_start = int(lag) - 2
    lag_end = int(lag) + 2
    if region == 1:
        new_latt = atlantic_latt.loc[lambda (atlantic_latt):atlantic_latt[0].astype(float)>=lat_start,:]
        new_latt = new_latt.loc[lambda (new_latt):new_latt[0].astype(float)<=lat_end,:]
        new_latt = new_latt.dropna(axis=1,how="all")

        new_lag = atlantic_long.loc[lambda (atlantic_long):atlantic_long[0].astype(float)>=lag_start,:]
        new_lag = new_lag.loc[lambda (new_lag):new_lag[0].astype(float)<=lag_end,:]
        new_lag = new_lag.dropna(axis=1,how="all") 
        
    elif region == 2:
        new_latt = pacific_latt.loc[lambda (pacific_latt):pacific_latt[0].astype(float)>=lat_start,:]
        new_latt = new_latt.loc[lambda (new_latt):new_latt[0].astype(float)<=lat_end,:]
        new_latt = new_latt.dropna(axis=1,how="all")

        new_lag = pacific_long.loc[lambda (pacific_long):pacific_long[0].astype(float)>=lag_start,:]
        new_lag = new_lag.loc[lambda (new_lag):new_lag[0].astype(float)<=lag_end,:]
        new_lag = new_lag.dropna(axis=1,how="all") 
    
    new_latt = new_latt.dropna(axis=1,how="all")
    new_lag = new_lag.dropna(axis=1,how="all")
    
    new_list.append(new_latt)
    new_list.append(new_lag)
    return new_list


# removing none value
def generate_new_prediction_data(data):
    data = data.dropna()
    return data

# after predicting each new data set is created. Scaling factor is 2.
def dataset_reform(lat_value,data,index):
    lat_start = int(lat_value) - 2
    lat_end = int(lat_value) + 2
    
    new_latt = data.loc[lambda (data):data[index].astype(float)>=lat_start,:]
    new_latt = new_latt.loc[lambda (new_latt):new_latt[index].astype(float)<=lat_end,:]
    
    return new_latt
    
def user_input():

    # taking input from user
    lat  = float(input("Enter the latitude"))
    lag  = float(input("Enter the longitude"))
    region = int(input("1 - Atlantic\n2 - Pacific\n"))    
    new_list = generate_new_dataset(lat,lag,region)

    lattData = pd.DataFrame(new_list[0])
    logData = pd.DataFrame(new_list[1])
    
    path_lat = []        
    path_log = []

   
    flag = True
    lat_col = 0
    lat_data = lattData[[lat_col,lat_col+1]]

    # implementing of rolling regression to lattitude and longitude data

    while(flag):
        t = generate_new_prediction_data(lat_data)
        lat = model_prep(t,lat,lat_col,lat_col+1)
        path_lat.append(lat[0])
        lat_col = lat_col + 1
        if lat_col == (len(lattData.columns) - 1):
            flag = False
        else:
            lat_data = dataset_reform(lat[0],lattData[[lat_col,lat_col+1]],lat_col) 
            lat_data = lat_data.dropna()
            if lat_data.empty:
                flag = False
    
    flag = True
    log_col = 0
    log_data = logData[[log_col,log_col+1]]
    while(flag):
        t = generate_new_prediction_data(log_data)
        lag = model_prep(t,lag,log_col,log_col+1)
        path_log.append(lag[0])
        log_col = log_col + 1
        if log_col == (len(logData.columns) - 1):
            flag = False
        else:
            log_data = dataset_reform(lag[0],logData[[log_col,log_col+1]],log_col) 
            log_data = log_data.dropna()
            if log_data.empty:
                flag = False
    
    
    path = []
    path.append(path_lat)
    path.append(path_log)
    return path



def get_path():
    # Dispaying predicted data
    path = user_input()
    print("\n\n\nLattitude Data : ")
    print(path[0])
    print("\n\n\nLongitude Data : ")    
    print(path[1])
    

    # combining data
    lat_df = pd.DataFrame(path[0])
    long_df = pd.DataFrame(path[1])
    df = pd.concat([lat_df, long_df], axis = 1)
    df = df.dropna()
    df.columns = ["latitude", "longitude"]
    
    # Ploting Data
    plt.plot(df['latitude'], df['longitude'], 'ro')
    plt.show()


get_path()





