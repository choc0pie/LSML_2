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

Service deployed as docker container.
I used async solution for server based on celery library.


Library requirments: 
- flask
- celery
- torch
- opencv-python
- torchvision

For successfull start you also need files with pretrained model and dictionary with labels:
- birdResNet.pt (https://drive.google.com/file/d/1-C5iD-IkrrYrzieYPJuEmCfgMXkxDmUS/view?usp=share_link)
- class_to_label.json

Reproduce container with Dockerfile:
1. docker build --tag python-docker .
2. docker run -d -p 8000:8000 python-docker


## API documentation
host: localhost
1. POST /api/v1/get_prediction
    - request data: encoded image with cv2.imencode
    - response data: json formatted with tag "task_id"
2. GET /api/v1/get_prediction/<task_id>
    - No request_data, task_id specified in endpoint
    - Response data: "status", and if prediction is ready - "result


