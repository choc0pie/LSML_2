from celery import Celery
from celery.result import AsyncResult
import time
from flask import Flask, request
import json
import torch
import numpy
import cv2
from torchvision import transforms, models

celery_app = Celery('server', backend='redis://localhost', broker='redis://localhost')
app = Flask('__name__')


def load_model(model_path):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features,2048),
                         nn.ReLU(),
                         nn.Dropout(0.3),
                         nn.Linear(2048,1024),
                         nn.ReLU(),
                         nn.Dropout(0.3),
                         nn.Linear(1024,450))
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model('birdResNet.pt')
f = open('class_to_label.json')
class_to_label = json.load(f)


@celery_app.task
def predict(img):
    with torch.no_grad():
        out = model(image)
    class_id = int(torch.max(out,dim=1)[1][0])
    result = class_to_label[str(class_id)]
    return result


@app.route('/api/v1/get_prediction', methods=["GET", "POST"])
def get_prediction_handler():
    if request.method == 'POST':
        r = request
        nparr = np.fromstring(r.data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.resize(img_tes,(224,224), interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        image = image
        convert_tensor = transforms.ToTensor()
        image = convert_tensor(image)
        image = torch.unsqueeze(image, 0)
       
        task = predict.delay(image) 
            
        response = {
            "task_id": task.id
        }
        return json.dumps(response)
    
    
@app.route('/api/v1/get_prediction/<task_id>')
def get_prediction_check_handler(task_id):
    task = AsyncResult(task_id, app=celery_app)
    if task.ready():
        response = {
            "status": "DONE",
            "result": task.result
        }
    else:
        response = {
            "status": "IN_PROGRESS"
        }
    return json.dumps(response)


if __name__ == "__main__":
    app.run('0.0.0.0', 8000)
