"""Optional offline SyncNet review; external model/code retain upstream license.

Scores are diagnostics, not a guarantee of natural faces or correct narration.
Uses the renderer's trusted JSON reference boxes, never unpickles user input.
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.signal import medfilt

parser=argparse.ArgumentParser()
parser.add_argument('--video',type=Path,required=True)
parser.add_argument('--coords',type=Path,required=True)
parser.add_argument('--syncnet-dir',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--offset-frames',type=int,default=0)
parser.add_argument('--start',type=float,default=0)
parser.add_argument('--duration',type=float,default=0)
args=parser.parse_args()
sys.path.insert(0,str(args.syncnet_dir))
sys.path.insert(0,str(args.syncnet_dir/'.deps'))
import python_speech_features
from SyncNetModel import S

torch.set_num_threads(4)
device='cuda' if torch.cuda.is_available() else 'cpu'
model=S().to(device).eval()
model.load_state_dict(torch.load(args.syncnet_dir/'data/syncnet_v2.model',map_location='cpu',weights_only=True))
boxes=np.asarray(json.loads(args.coords.read_text()),dtype=float)
if boxes.ndim!=2 or boxes.shape[1]!=4 or not len(boxes) or np.any(boxes[:,2:]<=boxes[:,:2]):
    raise ValueError('Invalid reference face coordinates')
cycle=np.concatenate([boxes,boxes[-2:0:-1] if len(boxes)>2 else boxes[::-1]])
bs=np.maximum(cycle[:,2]-cycle[:,0],cycle[:,3]-cycle[:,1])/2
xs=(cycle[:,0]+cycle[:,2])/2;ys=(cycle[:,1]+cycle[:,3])/2
bs=medfilt(bs,13);xs=medfilt(xs,13);ys=medfilt(ys,13)
cap=cv2.VideoCapture(str(args.video))
if abs(cap.get(cv2.CAP_PROP_FPS)-25)>.001: raise ValueError('SyncNet requires 25 fps')
start_frame=round(args.start*25)
n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-start_frame
if args.duration: n=min(n,round(args.duration*25))
cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
audio_cmd=['ffmpeg','-v','error','-ss',str(start_frame/25),'-i',str(args.video),'-t',str(n/25),'-vn','-ac','1','-ar','16000','-f','s16le','-']
audio=np.frombuffer(subprocess.check_output(audio_cmd),np.int16)
n=min(n,len(audio)//640)
mfcc=python_speech_features.mfcc(audio,16000).T
buffer=[]; read_count=0; vf=[];af=[]
with torch.inference_mode():
    for first in range(0,n-5,8):
        count=min(8,n-5-first)
        while read_count < first+count+4:
            ok,im=cap.read()
            if not ok: raise RuntimeError('Video decode stopped early')
            k=(start_frame+read_count+args.offset_frames)%len(cycle)
            size=bs[k];pad=math.ceil(size*1.5)
            im=np.pad(im,((pad,pad),(pad,pad),(0,0)),constant_values=110)
            x=xs[k]+pad;y=ys[k]+pad
            crop=im[int(y-size):int(y+size*1.5),int(x-size*1.25):int(x+size*1.25)]
            buffer.append(cv2.resize(crop,(224,224)))
            read_count+=1
        v=np.stack([np.stack(buffer[j:j+5],axis=0).transpose(3,0,1,2) for j in range(count)])
        a=np.stack([mfcc[:,(first+j)*4:(first+j)*4+20] for j in range(count)])[:,None]
        vf.append(model.forward_lip(torch.from_numpy(v).float().to(device)).cpu())
        af.append(model.forward_aud(torch.from_numpy(a).float().to(device)).cpu())
        buffer=buffer[count:]
cap.release()
vf=torch.cat(vf);af=torch.cat(af)

def score(v,a):
    # Exclude boundary padding instead of allowing zero vectors to bias lag.
    shift=15;length=len(v)
    if length < 2*shift+5: return None
    distances=[]
    for lag in range(-shift,shift+1):
        distances.append(torch.linalg.vector_norm(v[shift:length-shift]-a[shift+lag:length-shift+lag],dim=1).mean().item())
    best=int(np.argmin(distances));confidence=float(np.median(distances)-distances[best])
    return {'offset_frames':15-best,'offset_ms':(15-best)*40,'confidence':confidence,'distance':min(distances)}

windows=[]
for first in range(0,len(vf),250):
    value=score(vf[first:first+250],af[first:first+250])
    if value: windows.append({'start':start_frame/25+first/25,**value})
summary=score(vf,af)
result={'tool':'joonson/syncnet_python','reference_boxes':'renderer boxes; not independent face tracking',
        'global':summary,'windows':windows,'limitations':'Auxiliary metric, not perceptual or phoneme-level acceptance.'}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
