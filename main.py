import time
from flask import Flask, request
import json
import torch
from torch import nn
import numpy as np
import cv2
from torchvision import transforms, models

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

model = load_model('/app/birdResNet.pt')
f = open('/app/class_to_label.json')
class_to_label = json.load(f)


def predict(nparr):
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.resize(image,(224,224), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    image = image
    convert_tensor = transforms.ToTensor()
    image = convert_tensor(image)
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        out = model(image)
    class_id = int(torch.max(out,dim=1)[1][0])
    result = class_to_label[str(class_id)]
    return result

@app.route('/api/v1/get_prediction', methods=["GET", "POST"])
def get_prediction_handler():
    if request.method == 'POST':
        r = request
        nparr = np.frombuffer(r.data, np.uint8)
        result = predict(nparr) 

        response = {
        "result": result
        }       
        
        return json.dumps(response)


if __name__ == "__main__":
    app.run('0.0.0.0', 8000)
