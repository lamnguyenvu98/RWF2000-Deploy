import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, BackgroundTasks
import imageio.v3 as iio
import cv2
from math import ceil
from model import FGN
import torch
from collections import deque
from torchvision import transforms
from datasets.augmentation import Normalize, ToTensor
from utils import preprocessing
from copy import deepcopy
import base64

router = APIRouter(
    prefix='/inference',
    tags=['inference'],
    responses={404: {"description": "Not found"}},
)

tfms = transforms.Compose([
                    Normalize(),
                    ToTensor()])

classnames = ['Fight', 'NonFight']

queue = deque(maxlen=65)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = FGN()

def prediction(frame):
    if len(queue) <= 0: # At initialization, populate queue with initial frame
        for _ in range(64):
            queue.append(frame)
    # Add the read frame to last and pop out the oldest one
    queue.append(frame)
    queue.popleft()
    res = deepcopy(queue)
    res = preprocessing(frames=res)
    res = tfms(res)
    res = res.unsqueeze(0).permute(0, 4, 1, 2, 3).float()
    pred = model(res.to(device))
    best_idx = pred.softmax(-1).argmax(-1)
    score = pred.softmax(-1)[0][best_idx].item()
    
    label = classnames[best_idx]
    text = "{}: {:.1f}".format(label, score)
    # show_frame = queue.copy().pop()
    # show_frame = cv2.putText(show_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    # buffer = cv2.imencode('.jpg', show_frame)[1]
    # jpg_as_text = base64.b64encode(buffer)
    result = {
        "prob": score,
        "label": label,
        # "frames": jpg_as_text
    }
    return result

@router.on_event("startup")
async def load_checkpoint():
    model.load_from_checkpoint('checkpoints/model.ckpt').to(device)


@router.websocket('/predict')
async def predict(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            data = iio.imread(data, index=None, extension='.avi')

            for i in range(data.shape[0]):
                result = prediction(frame=data[i])
                result.update({ 
                    "percent_progress": ceil(i / data.shape[0] * 100) 
                })
                await websocket.send_json(result)
            cv2.destroyAllWindows()
    except WebSocketDisconnect:
        await websocket.close()