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
import datetime
import os
from fastapi.concurrency import run_in_threadpool
from concurrent.futures import ThreadPoolExecutor

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()

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
    
    pred_prob = pred.softmax(-1).detach().cpu().numpy().tolist()
    
    label = classnames[best_idx]
    text = "{}: {:.1f}".format(label, score)
    # show_frame = queue.copy().pop()
    # show_frame = cv2.putText(show_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    result = {
        "fight": pred_prob[0],
        "nonfight": pred_prob[1],
        "best_class": label,
    }
    return result

# def preprocessing_step(frame_lst):
#     frame = frame_lst.astype(dtype=np.float32)
#     if len(queue) <= 0: # At initialization, populate queue with initial frame
#         for _ in range(64):
#             queue.append(frame)
#     # Add the read frame to last and pop out the oldest one
#     queue.append()
#     queue.popleft()
#     res = deepcopy(queue)
#     res = preprocessing(frames=res)
#     res = tfms(res)
#     res = res.unsqueeze(0).permute(0, 4, 1, 2, 3).float()
#     return res.to(device)

# async def postprocessing_step(pred):
#     best_idx = pred.softmax(-1).argmax(-1)
#     # score = pred.softmax(-1)[0][best_idx].item()
#     pred_prob = pred.softmax(-1).detach().cpu().numpy().tolist()
#     result = {
#         "fight": pred_prob[0],
#         "nonfight": pred_prob[1],
#         "best_class": classnames[best_idx],
#     }
#     return result

async def predict_videos(data: np.ndarray, websocket: WebSocket) -> List[np.ndarray]:
    vid = []
    for i in range(data.shape[0]):
        result = {}
        # result = await prediction(frame=data[i])
        frame = data[i]
        _, buffer = cv2.imencode(".jpg", frame)
        # vid.append(frame)
        result.update({ 
            "frame_idx": i,
            "total_frame": data.shape[0],
        })
        await websocket.send_json(result)
        await websocket.send_bytes(buffer.tobytes())
    return vid

def run_predict(data, websocket):
    asyncio.run(predict_videos(data, websocket))

@router.websocket('/predict/{client_id}')
async def predict(websocket: WebSocket, client_id: int, background_task: BackgroundTasks) -> None:
    await manager.connect(websocket)
    try:
        while True:
            info = await websocket.receive_json()
            data = await websocket.receive_bytes()
            data = iio.imread(data, index=None, extension="." +  info['ext'])
            pool = ThreadPoolExecutor(max_workers=2)
            pool.submit(run_predict, data, websocket)
            
            # await run_in_threadpool(run_predict, data, websocket)
            
            # pool = ThreadPoolExecutor(max_workers=2)
            # preprocess_data = [data for data in pool.map(preprocessing_step, data.astype(np.object_))]
            # preds = [model(data) for data in preprocess_data]
            # results = [res for res in pool.map(postprocessing_step, preds)]
            
            # for frame_idx in range(data.shape[0]):   
            #     preprocess_data = preprocessing_step(data[frame_idx])
            #     pred = model(preprocess_data)
            #     pred_result = 
                # vid = await predict_videos(data, websocket)
            
            # background_task.add_task(write_video_s3, BUCKET, "test_videos/test", vid, ".mp4")
            # time = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
            # await write_video_s3(bucket=BUCKET, filename=os.path.join("predict", time + "_" + info["name"]), list_of_frames=vid)
            

    except WebSocketDisconnect:
        await websocket.close()
        manager.disconnect(websocket)
        manager.broadcast(f"Cam: {client_id} is done!")