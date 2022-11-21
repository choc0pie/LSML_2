# LSML_2

# Project: Bird classification

## Model
- Model: Fine-tuned ResNet50
- Dataset: https://www.kaggle.com/datasets/gpiosenka/100-bird-species
- Loss function: CrossEntropyLoss
- Optimizer: Adam optimizer with learning rate=0.0001

### Strategy
1. Loaded pretrained ResNet50
2. Added layers:
    - nn.Linear(model.fc.in_features,2048)
    - nn.ReLU()
    - nn.Dropout(0.3)
    - nn.Linear(2048,1024)
    - nn.ReLU()
    - nn.Dropout(0.3)
    - nn.Linear(1024,NUM_CLASSES)
 
3. Then model was trained for 10 epochs on 100-bird-species dataset.


Notebook with model training:  NartdinovKA_LSML_SGA_ResNetFineTuning.ipynb (Solved in Colab)

## Service deployment

Server deployed as docker container.
I used async solution for server based on celery library, because model pretty big and it might take a time for prediction.


Library dependences: 
- flask
- celery
- json
- torch
- cv2
- torchvision


- 
                        
