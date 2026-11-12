#!/usr/bin/env python3
"""Independent NumPy reference for the archived LeNet weights and CIFAR-10."""
import argparse,json,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--count',type=int,default=1000)
p.add_argument('--relu-range',type=float,default=10)
p.add_argument('--output',type=Path,default=ROOT/'reports/lenet/plain-numpy.json')
a=p.parse_args()
weights={f.stem:np.load(f).astype(np.float64) for f in (ROOT/'data/lenet_weights').glob('*.npy')}
raw=np.fromfile(ROOT/'data/cifar10/test_batch.bin',dtype=np.uint8).reshape(-1,3073)[:a.count]
x=raw[:,1:].reshape(-1,3,32,32).astype(np.float64)/255
x=(x-np.array([.4914,.4822,.4465])[None,:,None,None])/np.array([.2023,.1994,.2010])[None,:,None,None]
text=(ROOT/'include/extension/ReLUCoefficients.h').read_text()
coeff=[np.array([float.fromhex(s.strip()) for s in re.search('relu_p'+str(i)+r' = \{(.*?)\}',text).group(1).split(',')],dtype=np.longdouble) for i in [1,2,3]]
def conv(x,name):
 w,b=weights[name+'.weight'],weights[name+'.bias']
 windows=np.lib.stride_tricks.sliding_window_view(x,(5,5),axis=(-2,-1))
 return np.einsum('nchwkl,ockl->nohw',windows,w,optimize=True)+b[None,:,None,None]
def pool(x):
 n,c,h,w=x.shape
 return x.reshape(n,c,h//2,2,w//2,2).mean(axis=(3,5))
def dense(x,name):return x@weights[name+'.weight'].T+weights[name+'.bias']
records={}
for mode in ['exact','polynomial']:
 ranges=[]
 def activate(v):
  ranges.append({'min':float(v.min()),'max':float(v.max()),'outside':int((np.abs(v)>a.relu_range).sum())})
  if mode=='exact':return np.maximum(v,0)
  t=(v/a.relu_range).astype(np.longdouble)
  z=t
  for c in coeff:z=np.polynomial.polynomial.polyval(z,c)
  return np.asarray(a.relu_range*t*(.5+z),dtype=np.float64)
 y=pool(activate(conv(x,'conv1')))
 y=pool(activate(conv(y,'conv2'))).reshape(len(x),400)
 y=activate(dense(y,'fc1'));y=activate(dense(y,'fc2'));y=dense(y,'fc3')
 prediction=y.argmax(axis=1)
 records[mode]={'accuracy':float((prediction==raw[:,0]).mean()),'prediction_histogram':np.bincount(prediction,minlength=10).tolist(),'activation_ranges':ranges,'first_logits':y[0].tolist(),'predictions':prediction.tolist(),'logits':y.tolist()}
records['count']=len(x);records['relu_range']=a.relu_range
records['agreement']=float(np.mean(np.array(records['exact']['predictions'])==records['polynomial']['predictions']))
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps({m:{k:v for k,v in records[m].items() if k not in ['predictions','logits']} for m in ['exact','polynomial']},indent=2))
