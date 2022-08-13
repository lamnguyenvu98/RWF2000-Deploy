import asyncio
from typing import List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, BackgroundTasks
import imageio.v3 as iio
import cv2
import numpy as np
from model import FGN
import torch
from collections import deque
from torchvision import transforms
from datasets.augmentation import Normalize, ToTensor
from utils import preprocessing
from copy import deepcopy
from utils import write_video_s3
from io import BytesIO
import json

BUCKET = "rwf2000-bucket"

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

model = FGN().load_from_checkpoint('checkpoints/model.ckpt').to(device)

async def prediction(frame):
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
    show_frame = queue.copy().pop()
    show_frame = cv2.putText(show_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # result = {
    #     "prob": score,
    #     "label": label,
    # }
    return show_frame

async def predict_videos(data: np.ndarray, websocket: WebSocket) -> List[np.ndarray]:
    vid = []
    for i in range(data.shape[0]):
        # frame = await prediction(frame=data[i])
        frame = data[i]
        vid.append(frame)
        result = { 
            "frame_idx": i,
            "total_frame": data.shape[0],
            "is_finished": False
        }
        await websocket.send_json(result)
    return vid

@router.websocket('/predict')
async def predict(websocket: WebSocket, background_task: BackgroundTasks) -> None:
    await websocket.accept()
    try:
        while True:
            info = await websocket.receive_json()
            data = await websocket.receive_bytes()
            
            # data = iio.imread(data, index=None, extension='.avi')
            # res_data = 'OK' if data is not None else 'FAILED'
            # print(res_data)
            # await websocket.send_text(res_data)
            # vid = await predict_videos(data, websocket)
            # background_task.add_task(write_video_s3, BUCKET, "test_videos/test", vid, ".mp4")
            # await write_video_s3(bucket=BUCKET, filename="test_videos/test", list_of_frames=vid)

    except WebSocketDisconnect:
        await websocket.close()