import sys, os
import cv2, numpy as np
VIDEO = r"C:/Projects/now_agent_can/docs/assets/ghost-message.webm"
OUT = "inline_test"; os.makedirs(OUT, exist_ok=True)
def iter_gray(path, stride=1):
    cap = cv2.VideoCapture(path); i=0
    while True:
        ok,f=cap.read()
        if not ok: break
        if i%stride==0: yield cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        i+=1
    cap.release()
def accumulate(frames):
    dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    score=prev=prev_smooth=None; drift=np.zeros(2); pairs=0
    for gray in frames:
        if prev is not None:
            flow=dis.calc(prev,gray,None)
            bg=np.median(flow.reshape(-1,2),axis=0); residual=flow-bg
            mag=float(np.hypot(*bg))
            ps=(residual @ (-bg/mag)) if mag>0.15 else np.hypot(residual[...,0],residual[...,1])
            ps=np.clip(ps,0,None).astype(np.float32)
            smooth=cv2.GaussianBlur(ps,(31,31),0)
            if prev_smooth is not None:
                (dx,dy),r=cv2.phaseCorrelate(prev_smooth,smooth)
                if r>0.05 and np.hypot(dx,dy)<30: drift+=(dx,dy)
            prev_smooth=smooth
            h,w=ps.shape
            M=np.float32([[1,0,-drift[0]],[0,1,-drift[1]]])
            reg=cv2.warpAffine(ps,M,(w,h))
            score=reg if score is None else score+reg; pairs+=1
        prev=gray
    print(pairs,"frame pairs, drift",tuple(np.round(drift).astype(int)))
    return score
def reveal(score):
    score=np.clip(score,0,None); hi=np.percentile(score,99.5)
    norm=np.clip(score/hi*255,0,255).astype(np.uint8) if hi>0 else score.astype(np.uint8)
    norm=cv2.GaussianBlur(norm,(5,5),0)
    _,mask=cv2.threshold(norm,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k); mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,k)
    h_img=mask.shape[0]
    n,lab,st,_=cv2.connectedComponentsWithStats(mask)
    for i in range(1,n):
        x,y,w,h,area=st[i]
        streak = w>=5*h and h<=h_img//18
        if area<mask.size//20000 or streak: mask[lab==i]=0
    return norm,mask
score=accumulate(iter_gray(VIDEO)); heat,mask=reveal(score)
cv2.imwrite(os.path.join(OUT,"revealed.png"),mask)
n,lab,st,_=cv2.connectedComponentsWithStats((mask>127).astype('uint8'),8)
print("components:", n-1, ("-> streak removed, OK" if n-1==8 else "-> UNEXPECTED"))
