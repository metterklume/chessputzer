import numpy as np
import tensorflow as tf
from putzlib import boardtofen, splitboardcontour, pieces
import cv2
import PIL.Image as Image
import argparse
import glob
import imghdr

modlight = tf.keras.models.load_model('./saved_models/lightA.h5')
moddark  = tf.keras.models.load_model('./saved_models/darkA.h5')


class Board:
    def __init__(self, *impath):
        im = Image.open(impath[0])
        ima = np.asarray(im.convert("L"),dtype=np.uint8)
        self.squares = [np.bitwise_not(s) for s in splitboardcontour(ima)]

    def boardprednn2(self):
        if len(self.squares) != 64:
            return ""
        sqresized = [cv2.resize(im,(32,32),interpolation = cv2.INTER_CUBIC) for im in self.squares]
        sqdark = np.zeros((32,32,32,1),np.float32)
        sqlight = np.zeros((32,32,32,1),np.float32)
        for i,s in enumerate(sqresized):
            ld = (i + i//8)%2
            if ld == 0:
                sqlight[i//2,:,:,0] = (1/255.0)*np.float32(s)
            else:
                sqdark[i//2,:,:,0] = (1/255.0)*np.float32(s)
        predslight = np.argmax(modlight.predict(sqlight),axis=1)
        predsdark = np.argmax(moddark.predict(sqdark),axis=1)
        bb = ""
        for i in range(64):
            ld = (i + i//8)%2
            pred = (predslight,predsdark)[ld][i//2]
            bb += pieces[pred]
            if (i+1) % 8 ==0:
                bb+="/"
        return boardtofen(bb[:-1])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", required=False, help="Image file in png, jpeg or gif format")
    ap.add_argument("-o", "--output", required=False, default="fens.txt",
                    help="Output file. Default = fens.txt. Will be overwritten if it exists")
    ap.add_argument("-d", "--directory", required=False,
                    help="Image directory. All images in png, jpeg or gif format will be processed")
    args = vars(ap.parse_args())

    outfile = args.get("output", "fens.txt")
    if not args["file"] and not args["directory"]:
        print("Error. Either --file <imagefile> or --directory <imagedirectory> must be specified")
    if args["file"]:
        fname = args["file"]
        if not imghdr.what(fname):
            print("Not a valid image file")
        else:
            bd = Board(fname)
            fen = bd.boardprednn2()
            with open(outfile, "w") as fileout:
                fileout.write(fname + "\n")
                if fen:
                    fileout.write(fen + "\n")
                else:
                    fileout.write("Couldn't find chessboard\n")
    elif args["directory"]:
        imdir = args["directory"]
        if imdir[-1] != "/":
            imdir += "/"
        exts = ("png", "jpg", "gif")
        fnames = []
        for e in exts:
            fnames += glob.glob(imdir + "*" + e)
        with open(outfile, "w") as fileout:
            for fname in fnames:
                bd = Board(fname)
                fen = bd.boardprednn2()
                fileout.write(fname + "\n")
                if fen:
                    fileout.write(fen + "\n")
                else:
                    fileout.write("Couldn't find chessboard\n")
