# DP780 
## Dir Tree
```
├── data
│   ├── 590 
│   │     ├── train
│   │     │       └ [date]
│   │     │              ├ before
│   │     │              └ after
│   │     ├── val  
│   │     └── test
│   ├── 780
│   ├── 980
│   ├── 1180
│   ├── 590_corner.pkl
│   ├── 780_corner.pkl
│   ├── 980_corner.pkl
│   ├── 1180_corner.pkl
│   ├── MTS
│   ├── info.xlsx (strain rate for diff exp)
│   ├── properties.pkl (normalized features)
│   ├── properties.xlsx 
│   └── [model_acc]_[model_name]
│                              ├── data.pkl
│                              └── pred.pkl
├── clear_label
│             ├── 590.txt
│             ├── 780.txt
│             ├── 980.txt
│             ├── 1180.txt
│             └── label.py
├── model
│      └── [model_acc]_[model_name].pkl
├── func.py
├── func2.py
├── Class.py
├── deform.py
├── classify.py
├── test.ipynb
├── prediction.ipynb
└── edge.py
```

## 1. Add bounding boxes on deformed IPF
```
python deform.py [data_path]
```
For example

```
python deform.py 1180/val/
```
The data are under data/1180/val/

The IPF images will be dumped under output/1180/train/

<img src="https://i.imgur.com/Gs9swPR.png" height=200 
/> <img src="https://i.imgur.com/wOnMLQO.jpg" height=200 
/>

## 2. Data Labeling
1. Go to clear_label/, prepare [steel].txt
```
[date1]
x1,y1
x2,y2
[date2]
x1,y1
x2,y2
```
2. Convert xy to hw, save the result in data/[steel]_corner.pkl
```
python label.py [steel]
```
data/[steel]_corner.pkl
```
{[date]: list of [h,w]}
```
For example
```
{"20200916":  [[0, 250], [0, 299], ...],}
```

## 3. Model Training
prepare: properties.pkl, [steel]/, [steel]_corner.pkl

See train.sh for training command
```
python train.py [features joined by "_"]
```
For example
```
python train.py Error_Quaternion
```
Models with acc > 0.7 will be saved under model/

To check testing acc, see testing.ipynb
## 4. Phase Identification
```
python edge.py [data_path]
```
For example
```
python edge.py 1180/train/20210415/before/
```
The output cluster.pkl and some figures will be dumped under data/1180/train/20210415/before/

## 5. Phase Analysis and Crack Propagation
see prediction.ipynb

