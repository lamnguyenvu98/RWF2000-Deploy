import boto3
from fastapi import APIRouter, Response, Header, Request
import imageio.v3 as iio
from utils import s3_read
from io import BytesIO
from starlette.responses import StreamingResponse

BUCKET = 'rwf2000-bucket'

# s3 = boto3.resource('s3')
# filename = "/home/pep/drive/PCLOUD/Dataset/UCFCrime2Local/video-data/Arrest005_x264.mp4"
# s3.meta.client.upload_file(Filename = filename, Bucket = BUCKET, Key = 'arrest.mp4')

router = APIRouter(
    prefix='/video',
    tags=['video'],
    responses={404: {"description": "Not found"}},
)

@router.get("/result")
async def result(request: Request):
    data = s3_read(BUCKET, "test_videos/arrest002_x264.mp4")
    return StreamingResponse(BytesIO(data), media_type="video/mp4")

@router.get("/success_s3")
async def success_s3():
    pass