# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 16:37:09 2017

@author: klein
"""
from __future__ import division
import numpy as np
import PIL.Image as Image
import scipy.signal
import cv2
#from IPython.display import display
import glob
from os.path import basename
import requests
from io import BytesIO
import sys
import argparse,imghdr

#if sys.platform == 'win32':
#    import PIL.ImageGrab as ImageGrab

from putzlib import pathtemps,pieces,piecenames
from putzlib import houghbox,splitboardhough,splitboardcontour,boardtofen,norma

class Board():
    def __init__(self,*impath):
        if impath:
            self.im = Image.open(impath[0])
        else:
            self.im = ImageGrab.grabclipboard()
        self.ima = np.asarray(self.im.convert("L"),dtype=np.uint8)
        #self.imablur = np.uint8(gaussim(self.ima,m=4))
        self.squares = splitboardcontour(self.ima)
#        if not self.squares:
#            self.squares = splitboard(self.ima)
        self.squares = [np.bitwise_not(s) for s in self.squares]
        self.g = [-1]*64
    #
    #
    def getpieces(self,pbarrs):
        if not self.squares:
            return ""
        candidates = [[]]*64
        candprobs  = [[]]*64
        guesses = [-1]*64
        #guesspos = [0]*64
        #first pass
        for i,s in enumerate(self.squares):
            ld = (i + (i//8))%2
            tg,to = piecepred(s,pbarrs,ld)
            assert(len(tg)==len(to))
            to = [x for x in to if x>.5]
            tg = tg[len(tg)-len(to):]
            #
            assert(len(tg)==len(to))
            guesses[i] = tg[-1]
            candidates[i] = tg
            candprobs[i]  = to
        #
        piecemax=[8,2,2,2,1,1,8,2,2,2,1,1,64]
        def toomany(p,nextbest=True):
            inds = [i for i,q in enumerate(guesses) if q==p]
            if len(inds)<= piecemax[p]:
                return 0 #no changes needed
            for _ in range(piecemax[p]):
                m = np.argmax([candprobs[ind][-1] for ind in inds])
                inds.pop(m)
            if not nextbest:
                for q in inds:
                    guesses[q] = 12
            #
            else:
                for q in inds:
                    candidates[q].pop(-1)
                    candprobs[q].pop(-1)
                    if candidates[q]:
                        guesses[q] = candidates[q][-1]
                    else:
                        guesses[q] = 12
            print("changed %d %s"%(len(inds),pieces[p]))
            return len(inds)
        #        
        changed = 1
        while changed > 0:
            changed = 0 
            #no pawns on the first or eighth ranks
            for q in list(range(8))+list(range(56,64)):
                if guesses[q] == 0 or guesses[q]==6:
                    changed += 1
                    candidates[q].pop(-1)
                    candprobs[q].pop(-1)
                    if candidates[q]:
                        guesses[q] = candidates[q][-1]
                    else:
                        guesses[q] = 12
            #one king each
            changed += toomany(5,nextbest=True)
            changed += toomany(11,nextbest=True)
            #one queen each
            changed += toomany(4,nextbest=True)
            changed += toomany(10,nextbest=True)
            #two rooks
            changed += toomany(7,nextbest=True)
            changed += toomany(1,nextbest=True)
            #two knights
            changed += toomany(8,nextbest=True)
            changed += toomany(1,nextbest=True)
            #eight pawns
            changed += toomany(6,nextbest=True)
            changed += toomany(0,nextbest=True)
            #two bishops
            changed += toomany(9,nextbest=True)
            changed += toomany(3,nextbest=True)
            #
        self.g = guesses
        self.b=""
        for i,s in enumerate(guesses):
            if i%8 == 0 and i>0:
                self.b+='/'
            ld = (i + (i//8))%2 #light or dark square
            self.b+= pieces[guesses[i]]
        return boardtofen(self.b)

def piecepred(imarr,pset,ld):
    nsets = len(pset)//12
    if ld == 1 and stripetest(imarr) > .15: #screen out blank images
        return [12],[1]
    if ld == 0 and blanktest(imarr) < 10:   #screen out blank images
        return [12],[1]
#    imarr=np.where(imarr<100,0,imarr)
    if ld == 1: #get rid of some of the striped background & resize
        imarr = backfilter(imarr)
    else:      #just resize
        imarr = backfilter(imarr)
        #imarr = cv2.resize(imarr,(32,32),interpolation=cv2.INTER_AREA)
    imarr = np.float32(imarr)
    if np.mean(imarr)<1:
        return [12],[1]
    imarr = imarr/norma(imarr)
    overs = [overlap(p,imarr) for p in pset]
    if max(overs) < .55: #not enough overlap with any piece
        return [12],[1]
    redovers = [max(overs[i:i+nsets]) for i in range(0,len(overs),nsets)]
    topguesses = np.argsort(redovers)[-4:]
    topovers   = [redovers[i] for i in topguesses]
    return list(topguesses),list(topovers)

def overlap(imarr1,imarr2):
	imarr1,imarr2 = np.float32(imarr1),np.float32(imarr2)
	overs = scipy.signal.convolve2d(imarr1,imarr2[::-1,::-1],mode='valid')
	return overs.max()

def stripefilter(ima):
    l,w = ima.shape
    imf = np.fft.fft2(ima)
    for j in range(2,int(l/2)):
        imf[j,j],imf[j,j+1],imf[j+1,j]=0,0,0
        imf[l-j,w-j],imf[l-j,w-j-1],imf[l-j-1,w-j]=0,0,0
    ima2 = np.real(np.fft.ifft2(imf))
    ima2 = np.clip(ima2,0,255)
    return ima2.astype(np.uint8)

def pieceacc(pathtodir,piece,units,ld):
    pnames = glob.glob(pathtodir+piecenames[piece]+"*")
    if not pnames:
        return (-1,[],[])
    pims = [Image.open(pname) for pname in pnames]
    parrs = [np.bitwise_not(np.asarray(im,np.uint8)) for im in pims]
    results = [piecepred(imarr,units,ld)[0][-1] for imarr in parrs]
    corr = [1 if i==pieces.index(piece) else 0 for i in results]
    #corr2 = [1 if (i-pieces.index(piece))%6==0 else 0 for i in results]
    wrongs = [results[i] for i,x in enumerate(corr) if x==0]
    wrongnames = [pnames[i] for i,x in enumerate(corr) if x==0]
    return sum(corr)/len(corr),wrongs,wrongnames


def blanktest(ima):
    '''return max of the central half of the ima'''
    l,w = ima.shape
    m = np.max(ima[int(l/4):int(3*l/4),int(w/4):int(3*w/4)])
    return m

def stripetest(ima):
    '''check for stripe pattern in fft'''
    imf = np.fft.fft2(ima)
    x = -1
    for i in (-1,0,1):
        y = np.max(np.abs(imf.diagonal(i)[1:]))**2/np.abs(imf[0,0])**2
        x = max(x,y)
    return x

def backfilter(im,finalshape=(32,32),passes=2):
    #imcvg= cropim(im,margin=.08)
    imcvg=im
    imcvg= cv2.resize(imcvg,(128,128),interpolation = cv2.INTER_CUBIC)
    #imcvg = cv2.GaussianBlur(imcvg,(11,11),0)
    for _ in range(passes):
        mask =np.zeros(imcvg.shape,np.uint8)
        ee = cv2.Canny(imcvg,200,400,apertureSize = 3)
        _,contours,_ = cv2.findContours(ee, 1, 2)
        dashes = [c for c in contours if abs(cv2.minAreaRect(c)[2]+45)<8]
        cv2.drawContours(mask,dashes,-1,255,4)
        imclean = np.where(mask==0,imcvg,0)
        imcvg = imclean
    return cv2.resize(imclean,finalshape,interpolation=cv2.INTER_AREA)

def fentoimg(fen):
    fenurl='http://www.fen-to-image.com/image/'+fen
    response = requests.get(fenurl)
    img = Image.open(BytesIO(response.content))
    return img

### Load image templates
nsets = 3
alltemps = np.load('pbarrs.npz').items()[0][1]
numpieces = nsets * 12
rows = alltemps.shape[0]//numpieces
pbarrs = [alltemps[rows*i:rows*(i+1)][:] for i in range(numpieces)]
assert(len(pbarrs) == numpieces)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()    
    ap.add_argument("-f", "--file", required=False,help="Image file in png, jpeg or gif format")
    ap.add_argument("-o","--output",required=False,default="fens.txt",
                    help="Output file. Default = fens.txt. Will be overwritten if it exists")
    ap.add_argument("-d","--directory",required=False,help="Image directory. All images in png, jpeg or gif format will be processed")    
    args = vars(ap.parse_args())
    
    outfile=args.get("output","fens.txt")
    if not args["file"] and not args["directory"]:
        print("Error. Either --file <imagefile> or --directory <imagedirectory> must be specified")
    if args["file"]:
        fname = args["file"]
        if not imghdr.what(fname):
            print("Not a valid image file")
        else:
            bd = Board(fname)
            fen = bd.getpieces(pbarrs)
            with open(outfile,"w") as fileout:
                fileout.write(fname+"\n")
                if fen:
                    fileout.write(fen+"\n")
                else:
                    fileout.write("Couldn't find chessboard\n")
    elif args["directory"]:
        imdir = args["directory"]
        if imdir[-1] != "/":
            imdir += "/"
        exts = ("png","jpg","gif")
        fnames = []
        for e in exts:
            fnames += glob.glob(imdir+"*"+e)
        with open(outfile,"w") as fileout:
            for fname in fnames:
                bd = Board(fname)
                fen = bd.getpieces(pbarrs)
                fileout.write(fname+"\n")
                if fen:
                    fileout.write(fen+"\n")
                else:
                    fileout.write("Couldn't find chessboard\n")

