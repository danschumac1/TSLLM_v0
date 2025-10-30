# DO IT ALL TIME SERIES

## 1. Get data

### Classification Datasets

#### CPU
- Open: https://www.timeseriesclassification.com/description.php?Dataset=Computers  
- Download the dataset archive (zip). (click **Download this dataset**)
- Extract
- contents should go into `./raw_data/cpu/`
- ensure that Computers_TRAIN.txt & Computers_TEST.txt are in `./raw_data/cpu/`

#### EMG
- Open: https://physionet.org/content/emgdb/1.0.0/
- click **Download the ZIP file (5.1 MB)**
- extract and put contents into `./raw_data/emg`

#### ECG
- Open: https://physionet.org/content/challenge-2017/1.0.0/  
- click **Download the ZIP file (1.4 GB)**
- extract sample2017 and put contents into `./raw_data/sample2017/`
- extract training2017 and put contents into `./raw_data/training2017/`

#### HAR
- Open: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- click **DOWNLOAD (58.2 MB)**
- extract and put contents into `./raw_data/har/`

#### RWC
- Open: https://www.kaggle.com/competitions/whale-detection-challenge/data
- Login and download
- extract and put contents into ./raw_data/rwc

#### TEE
- Open: https://www.timeseriesclassification.com/description.php?Dataset=Lightning7  
- Download the dataset archive (zip).
- Extract and put contents into `./raw_data/tee/`

## 2. Clean/preprocess the data
- `chmod +x ./bin/clean_all_data.sh`
- `./bin/clean_all_data.sh`