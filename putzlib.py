from __future__ import division
import os
import numpy as np
import PIL.Image as Image
import scipy.signal
import cv2
import PIL.ImageFilter as ImageFilter
try:
    from IPython.display import display
except ImportError:
    pass
#import matplotlib.pyplot as plt
import glob
from io import BytesIO
from math import floor
from numpy.linalg import norm

# pathboards = os.path.expanduser("~")+"/Dropbox/putz/boards3/"
# pathlight = pathboards+'lightsquares/'
# pathdark = pathboards + 'darksquares/'
# pathavglight = pathboards+'pieceaverages/'+'lightsquares/'
# pathavgdark =  pathboards+'pieceaverages/'+'darksquares/'
pathtemps = os.path.expanduser("~")+"/Dropbox/putz/piecetemps/"

pieces=['P','R','N','B','Q','K','p','r','n','b','q','k',' ']
piecenames={' ':'zz'}
for p in pieces[:6]:
    piecenames[p] = 'w'+p.lower()
for p in pieces[6:12]:
    piecenames[p] = 'b'+p.lower()
piecenamesrev={}
for k,v in piecenames.items():
    piecenamesrev[v] = k

def nvec(v):
    a = v.dot(v)
    return v/np.sqrt(a)

def norma(arr):
    arr2=np.float32(arr.flatten())
    return np.sqrt(arr2.dot(arr2))

def skel(a):
    acop = np.copy(a)
    l = a.shape[0]
    for i in range(l-1):
        if acop[i] < acop[i+1]:
            acop[i]=0
    for i in range(l-1,1,-1):
        if acop[i] <= acop[i-1]:
            acop[i]=0
    return acop

def displayarray(a,rng=[0,255]):
    ma,mi = rng[1],rng[0]
    a = ((a-mi)/(ma-mi))*255
    a = np.clip(a,0,255)
    return Image.fromarray(np.uint8(a))

def gaussim(a,m=6,sdev=2):
    f = scipy.signal.gaussian(m,2)
    gwin = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            gwin[i,j] = f[i]*f[j]
    return scipy.signal.convolve2d(a,gwin/gwin.sum(),mode='same')

def probs(overs,nsets=3):
    pl = np.array([max(overs[i:i+nsets]) for i in range(0,len(overs),nsets)])
    return [(pieces[i],pl[i]) for i in np.argsort(pl)[::-1][:4]]

def piecepredold(imarr,pset,ld):
    nsets = len(pset)/12
    imarr=np.where(imarr<100,0,imarr)
    if ld == 1: #get rid of some of the background and resize
        imarr = backfilter(imarr)
    else: #just resize
        imarr = cv2.resize(imarr,(32,32),interpolation=cv2.INTER_AREA)
    imarr = np.float32(imarr)
    if np.mliensean(imarr)<1:
        return 12
    imarr = imarr/norma(imarr)
    overs = [overlap(p,imarr) for p in pset]
    if max(overs) < .5:
        return 12
    else:
        return int(np.argmax(overs)/nsets)

def piecepredebug(imarr,pset,ld):
    nsets = len(pset)/12
#    if blankpred(imarr,ld) == 1:
#        return 12
    imarr=np.where(imarr<100,0,imarr)
    if ld == 1: #get rid of some of the background and resize
        imarr = backfilter(imarr)
    else: #just resize
        imarr = cv2.resize(imarr,(32,32),interpolation=cv2.INTER_AREA)
    imarr = np.float32(imarr)
    if np.mean(imarr)<1:
        print(empty)
        return 12,[]
    imarr = imarr/norma(imarr)
    overs = [overlap(p,imarr) for p in pset]
    if max(overs) < .4:
        print('max too low')
        return 12,overs
    else:
        return int(np.argmax(overs)/nsets),overs

def pprobs(ima,ld=1):
    _,ov=piecepredebug(ima,pbarrs,ld)
    return probs(ov)

def boardtofen(board):
    board = board.split('/')
    fen = ''
    for row in board:
        count=0
        for p in row:
            if p in pieces[:-1]:
                if count>0:
                    fen+=str(count)
                fen += p
                count=0
            else:
                count+=1
        if count>0:
            fen+=str(count)
        fen += '/'
    return fen[:-1]

def houghbox(imarr,minfactor=.4,rho=1):
    l,w = imarr.shape
    tol = np.pi/20
    edges = cv2.Canny(imarr,100,200,apertureSize = 3)
    lines = cv2.HoughLines(edges,rho,np.pi/50,int(minfactor*l))
    if lines is None:
        return [],[],(0,0),(0,0)
    #careful below! I had horizontal and vertical confused at first
    #and of course it still works if the image is almost a square
    vlines = np.sort([x[0] for [x] in lines if abs(x[1])<tol]).astype(np.int)
    hlines = np.sort([x[0] for [x] in lines if abs(x[1]-np.pi/2)<tol]).astype(np.int)
    if len(vlines) <=1  or len(hlines) <= 1:
        print("Not enough lines")
        return hlines,vlines,(0,0),(0,0)
    up,down = 0,len(hlines)-1
    #print('check')
    while hlines[down]-hlines[up] > l*7/8:
        up+=1
    up-=1
    while hlines[down]-hlines[up] > l*7/8:
        down-=1
    if down<len(hlines)-1:
        down+=1
    #
    left,right = 0,len(vlines)-1
    while vlines[right]-vlines[left] > w*7/8:
        left+=1
    left-=1
    while vlines[right]-vlines[left] > w*7/8:
        right-=1
    if right<len(vlines)-1:
        right+=1
    #print(hlines,up,down,vlines,left,right)
    return hlines,vlines,(hlines[up],hlines[down]),(vlines[left],vlines[right])


def splitboardhough(ima,minfactor=.33):
    l,w = ima.shape
    imablur = np.uint8(gaussim(ima,m=5))
    _,_,(up,down),(left,right) = houghbox(imablur,minfactor=minfactor)
    if down-up< l*3/4 or right-left < w*3/4:
        return []
    else:
        xw,yw = int(round((down-up)/8)),int(round((right-left)/8))
        print(xw,yw)
        return [ima[up+j*xw:up+(j+1)*xw,left+i*yw:left+(i+1)*yw] for j in range(8) for i in range(8)]

def splitboardcontour(ima):
    bounds = contourbox(ima)
    if bounds is None:
        imapad = np.pad(ima,(2,),'constant',constant_values=(0,))
        imapad = np.pad(imapad,(2,),'constant',constant_values=(255,))
        bounds = contourbox(imapad)
        ima = imapad
    if bounds:
        up,down,left,right = bounds
        xw,yw = int(round((down-up)/8)),int(round((right-left)/8))
        print(xw,yw)
        return [ima[up+j*xw:up+(j+1)*xw,left+i*yw:left+(i+1)*yw] for j in range(8) for i in range(8)]
    else:
        return []

def contourbox(ima):
    l,w = ima.shape
    for k in (11,7,3):
        imablur = cv2.GaussianBlur(ima,(k,k),0)
        ee = cv2.Canny(imablur,200,400,apertureSize = 5)
        contours,_ = cv2.findContours(ee, 1, 2)
        boxes = [c for c in contours if cv2.contourArea(c)>.8*l*w]
        if boxes:
            break
    if not boxes:
        ee = cv2.Canny(ima,100,200,apertureSize = 5)
        contours,_ = cv2.findContours(ee, 1, 2)
        boxes = [c for c in contours if cv2.contourArea(c)>.8*l*w]
    if not boxes:
        kernel = np.ones((5,5),np.uint8)
        opening = cv2.morphologyEx(ima,cv2.MORPH_OPEN, kernel)
        for k in (11,7,3):
            imablur = cv2.GaussianBlur(opening,(k,k),0)
            ee = cv2.Canny(imablur,100,200,apertureSize = 5)
            contours,_ = cv2.findContours(ee, 1, 2)
            boxes = [c for c in contours if cv2.contourArea(c)>.8*l*w]
            if boxes:
                break
    # if not boxes:
    #     ee = cv2.Canny(imapad,100,200,apertureSize = 5)
    #     _,contours,_ = cv2.findContours(ee, 1, 2)
    #     boxes = [c for c in contours if cv2.contourArea(c)>.8*l*w]
    if not boxes:
        return None
    areas = [cv2.contourArea(b) for b in boxes]
    arcareas = [(cv2.arcLength(b,True)/4)**2 for b in boxes]
    sqboxes = [b for i,b in enumerate(boxes) if (1-areas[i]/arcareas[i])<.1]
    if sqboxes:
        sqareas = [cv2.contourArea(b) for b in sqboxes]
        inbox = sqboxes[np.argmin(sqareas)]
    else:
        return None
    inboxflat = inbox.reshape(inbox.shape[0],-1)
    #
    x,y,w,h = cv2.boundingRect(inbox)
    tl,tr,br,bl = (x,y),(x+w,y),(x+w,y+h),(x,y+h)
    rect = np.array([tl,tr,br,bl])
    corners = [np.argmin([norm(inboxflat[j]-pt) for j in range(inboxflat.shape[0])])
           for pt in rect]
    toparc = subarc(inboxflat,corners[0],corners[1],corners[2])[:,1]
    bottomarc = subarc(inboxflat,corners[2],corners[3],corners[0])[:,1]
    leftarc = subarc(inboxflat,corners[0],corners[3],corners[1])[:,0]
    rightarc = subarc(inboxflat,corners[1],corners[2],corners[0])[:,0]
    #
    (t,b,l,r)=[int(round(np.median(a))) for a in (toparc,bottomarc,leftarc,rightarc)]
    return t,b,l,r

def showboardlines(ima,thickness=3):
    sides = contourbox(ima)
    if not sides:
        return None
    t,b,l,r = sides
    imacol = cv2.cvtColor(ima,cv2.COLOR_GRAY2BGR)
    cv2.line(imacol,(l,t),(r,t),(0,0,255),thickness)
    cv2.line(imacol,(l,b),(r,b),(0,0,255),thickness)
    cv2.line(imacol,(l,t),(l,b),(0,0,255),thickness)
    cv2.line(imacol,(r,t),(r,b),(0,0,255),thickness)
    return imacol

def subarc(arc,start,end,other):
    arcf = arc.reshape(-1,2)
    arclen = arcf.shape[0]
    start,end = min(start,end),max(start,end)
    if end-start<2:
        return arc[start:end+1]
    if start < other and other < end:
        return np.concatenate((arc[end:],arc[:start+1]))
    else:
        return arc[start:end+1]

def subarc2(arc,start,end):
    arcf = arc.reshape(-1,2)
    arclen = arcf.shape[0]
    if abs(start-end)<arclen//2:
        if start < end:
            return arc[start:end+1]
        else:
            return arc[end:start+1][::-1]
    else:
        if start < end:
            return np.concatenate((arc[end:],arc[:start+1]))[::-1]
        else:
            return np.concatenate((arc[start:],arc[:end+1]))

def splitboard(ima):
    fudge = -2
    l,w = ima.shape
    xl,yl,_,_ = findlines(ima)
    xlines,ylines = findlineset(xl,10),findlineset(yl,10)
    if len(xlines)<2 or len(ylines)<2:
        return [] #not enough lines
    xwidth,ywidth = int(np.mean(np.diff(xlines))),int(np.mean(np.diff(ylines)))
    if abs(xwidth*8 - w) > xwidth or abs(ywidth*8 - l)>ywidth:
        return [] #lines too far apart/too close together to be one square
    xstart = max(0,xlines[0]-floor((fudge+xlines[0])/xwidth)*xwidth)
    ystart = max(0,ylines[0]-floor((fudge+ylines[0])/ywidth)*ywidth)
    if xstart+7*xwidth > w:
        np.pad(ima,(0,0,0,xstart+7*xwidth-w),mode='edge')
    if ystart+7*ywidth > l:
        np.pad(ima,(0,ystart+7*ywidth-l,0,0),mode='edge')
    xlines = [xstart + n*xwidth for n in range(8)]
    ylines = [ystart + n*ywidth for n in range(8)]
    #print(xlines,ylines)
    squares= [ima[j:j+xwidth,i:i+ywidth] for j in ylines for i in xlines]
    return squares

def showbounds(ima,minfactor=.4):
    imc = np.copy(ima)
    imc = np.uint8(gaussim(ima,m=5))
    hlines,vlines,(up,down),(left,right)=houghbox(imc,minfactor)
    plt.imshow(imc)
    for x in vlines:
        plt.axvline(x,color='b')
    for y in hlines:
        plt.axhline(y,color='r')
    return None

def findlines(ima):
    ima = gaussim(ima)
    # find x and y gradients
    xkern = [[-1.,0., 1.],[-1.,0., 1.],[-1.,0., 1.]]
    ykern = [[1.,1.,1.],[0.,0.,0.],[-1.,-1.,-1.]]
    imax = scipy.signal.convolve2d(ima,xkern)
    imay = scipy.signal.convolve2d(ima,ykern)
    #Use the cool trick from chessfenbot. Lines are alternating.
    imax_pos = np.clip(imax,0,600)
    imax_pos = np.sum(imax_pos,0)
    imax_neg = np.clip(imax,-600,0)
    imax_neg = np.sum(imax_neg,0)
    # y lines
    imay_pos = np.clip(imay,0,600)
    imay_pos = np.sum(imay_pos,1)
    imay_neg = np.clip(imay,-600,0)
    imay_neg = np.sum(imay_neg,1)
    # normalize and return
    tt=(imax_pos*np.abs(imax_neg))
    tt=tt/tt.max()
    rr=(imay_pos*np.abs(imay_neg))
    rr=rr/rr.max()
    xlines = np.where(skel(tt)>.3)[0]
    ylines = np.where(skel(rr)>.3)[0]
    return xlines,ylines,tt,rr

def findlineset(lines,mindiff):
    #kludgy
    l = len(lines)
    count,start,dmax = 1,0,mindiff
    linesmax=[]
    for i in range(l-4):
        for j in range(i+1,min(i+4,l)):
            ltries = [lines[i]]
            d = lines[j]-lines[i]
            k = j
            if d > mindiff:
                restdiff = [abs(lines[r]-ltries[-1]-d) for r in range(k,l)]
                while min(restdiff) < 5:
                    ltries += [lines[k+np.argmin(restdiff)]]
                    k = np.argmin(restdiff)+1
                    restdiff = [abs(lines[r]-ltries[-1]-d) for r in range(k,l)]
            if len(ltries)>count:
                count,start,dmax,linesmax = len(ltries),i,d,ltries
    return linesmax

def piecetemp(pathbench):
    benchfiles=[glob.glob(pathbench+piecenames[p]+'*png') for p in pieces[:12]]
    ptemps = [Image.open(p).convert('L').resize((32,32),Image.ANTIALIAS) for pname in benchfiles for p in pname]
    pbarrs = [np.bitwise_not(np.asarray(p,np.uint8)) for p in ptemps]
    pbarrs = [np.float32(arr)/norma(arr) for arr in pbarrs]
    pbarrs = [np.pad(arr,((8,8),(8,8)),mode='constant') for arr in pbarrs]
    return pbarrs

def cropim(imarr,margin=.1):
    ima = np.copy(imarr)
    assert(margin<1 and margin > 0)
    l,w = ima.shape
    cmargin=1-margin
    ima[:int(margin*l)]=ima[int(cmargin*l):]=0
    ima[:,:int(margin*w)]=ima[:,int(cmargin*w):]=0
    return ima

def predfile(fpath,ld):
    imarr = np.asarray(Image.open(fpath),np.uint8)
    imarr = np.bitwise_not(imarr)
    return piecepred(imarr,pbarrs,ld)

def imagerow(ims,finshape=(256,256)):
    resized = [cv2.resize(im,finshape,interpolation = cv2.INTER_CUBIC) for im in ims]
    comb = np.zeros((finshape[0],finshape[1]*len(ims)),np.uint8)
    for i,im in enumerate(resized):
        comb[:,i*finshape[0]:(i+1)*finshape[0]] = im
    return comb

def imagerowborder(ims,finshape=(256,256),bord=4,color=(255, 127, 80)):
    uniform = [cv2.resize(im,finshape,interpolation = cv2.INTER_CUBIC) for im in ims]
    uniform = [cv2.cvtColor(im,cv2.COLOR_GRAY2BGR) for im in uniform]
    bordered = [cv2.copyMakeBorder(im, bord,bord,bord,bord, cv2.BORDER_CONSTANT, value=color) for im in uniform]
    dims = (bordered[0].shape[0],len(ims)*bordered[0].shape[1],3)
    comb = np.zeros(dims,np.uint8)
    for i,im in enumerate(bordered):
        comb[:,i*dims[0]:(i+1)*dims[0],:] = im
    return comb
