from fastapi import FastAPI, Body, UploadFile, File, Request, WebSocket
from fastapi.templating import Jinja2Templates
from starlette.responses import JSONResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect
import imageio.v3 as iio
import cv2
from routers import inference, get_video
import uvicorn

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*']
    )
]

app = FastAPI(middleware=middleware)

templates = Jinja2Templates(directory="templates")

@app.get("/")
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

app.include_router(inference.router)
app.include_router(get_video.router)

# if __name__ == '__main__':
#     uvicorn.run(app, port=8000, reload=True)