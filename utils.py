from typing import List
import cv2
import numpy as np
import boto3
import imageio as iio

def getOpticalFlow(video):
    """Calculate dense optical flow of input video
    Args:
        video: the input video with shape of [frames,height,width,channel]. dtype=np.array
    Returns:
        flows_x: the optical flow at x-axis, with the shape of [frames,height,width,channel]
        flows_y: the optical flow at y-axis, with the shape of [frames,height,width,channel]
    """
    # initialize the list of optical flows
    gray_video = []
    for i in range(len(video)):
        img = cv2.cvtColor(video[i], cv2.COLOR_RGB2GRAY)
        gray_video.append(np.reshape(img,(224,224,1)))

    flows = []
    for i in range(0,len(video)-1):
        # calculate optical flow between each pair of frames
        flow = cv2.calcOpticalFlowFarneback(gray_video[i], gray_video[i+1], None, 0.5, 3, 15, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        # subtract the mean in order to eliminate the movement of camera
        flow[..., 0] -= np.mean(flow[..., 0])
        flow[..., 1] -= np.mean(flow[..., 1])
        # normalize each component in optical flow
        flow[..., 0] = cv2.normalize(flow[..., 0],None,0,255,cv2.NORM_MINMAX)
        flow[..., 1] = cv2.normalize(flow[..., 1],None,0,255,cv2.NORM_MINMAX)
        # Add into list 
        flows.append(flow)
        
    # Padding the last frame as empty array
    flows.append(np.zeros((224,224,2)))
      
    return np.array(flows, dtype=np.float32)

def dynamic_crop(video):
    # extract layer of optical flow from video
    opt_flows = video[...,3]
    # sum of optical flow magnitude of individual frame
    magnitude = np.sum(opt_flows, axis=0)
    # filter slight noise by threshold 
    thresh = np.mean(magnitude)
    magnitude[magnitude<thresh] = 0
    # calculate center of gravity of magnitude map and adding 0.001 to avoid empty value
    x_pdf = np.sum(magnitude, axis=1) + 0.001
    y_pdf = np.sum(magnitude, axis=0) + 0.001
    # normalize PDF of x and y so that the sum of probs = 1
    x_pdf /= np.sum(x_pdf)
    y_pdf /= np.sum(y_pdf)
    # randomly choose some candidates for x and y 
    x_points = np.random.choice(a=np.arange(224), size=10, replace=True, p=x_pdf)
    y_points = np.random.choice(a=np.arange(224), size=10, replace=True, p=y_pdf)
    # get the mean of x and y coordinates for better robustness
    x = int(np.mean(x_points))
    y = int(np.mean(y_points))
    print('x-mean:', x)
    print('y-mean:', y)
    # avoid to beyond boundaries of array
    x = max(56,min(x,167))
    y = max(56,min(y,167))
    print('x:', x)
    print('y:', y)
    # get cropped video 
    return video[:,x-56:x+56,y-56:y+56,:]

def preprocessing(frames, dynamic_crop=False):
    v = []
    for i in range(len(frames)):
        frame_d = cv2.resize(frames[i].copy(), (224, 224), interpolation=cv2.INTER_AREA)
        frame_d = cv2.cvtColor(frame_d, cv2.COLOR_BGR2RGB)
        frame_d = np.reshape(frame_d, (224,224,3))
        v.append(frame_d)
    
    v = np.array(v)
    flows = getOpticalFlow(v)

    result = np.zeros((len(flows), 224, 224, 5))
    result[...,:3] = v
    result[...,3:] = flows
    
    if dynamic_crop:
        crop = dynamic_crop(result)
        crop_rs = np.zeros((crop.shape[0], 224, 224, 5))
        for i in range(crop.shape[0]):
            crop_rs[i] = cv2.resize(crop[i], (224, 224))
        
        return crop_rs
    
    return result

def s3_read(bucket, filename):
    s3 = boto3.client('s3')
    key = filename
    print("Requesting object from Bucket: {} and Key: {}".format(bucket, key))
    obj = s3.get_object(Bucket=bucket, Key=key)
    print("Got object from S3")
    return obj['Body'].read()

def write_video_s3(bucket: str, filename: str, list_of_frames: List[np.ndarray], extension: str = ".mp4"):
    filename = filename + extension
    vid = np.stack([x for x in list_of_frames])
    encoded = iio.v3.imwrite("<bytes>", vid, extension=extension, plugin="pyav", fps=30, codec="h264")
    s3 = boto3.client('s3')
    s3.put_object(Bucket = bucket, Key = filename, Body = encoded)
    print('Put file "{}" to bucket "{}" succesful'.format(bucket, filename))

if __name__ == '__main__':
    # s3_read("rwf2000-bucket", "arrest.mp4")
    # s3 = boto3.client('s3')
    # url = s3.generate_presigned_url( ClientMethod='get_object', Params={ 'Bucket': "rwf2000-bucket", 'Key': "arrest.mp4" } )
    
    video_path = '/home/pep/drive/PCLOUD/Dataset/UCFCrime2Local/video-data/Arrest002_x264.mp4'

    vid = []
    
    for idx, frame in enumerate(iio.imiter(video_path)):
        vid.append(frame)
    

    write_video_s3(bucket="rwf2000-bucket", filename="test_videos/arrest002_x264",
                   list_of_frames=vid, extension=".mp4")
    
    ######### WEBCAM #######################
    # for idx, frame in enumerate(iio.imiter("<video0>")):
    #     print(frame.shape)
    #     break

