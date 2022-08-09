#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 16:45:10 2022

@author: joe
"""

from PIL import Image,ImageOps
import numpy as np

binarise_thresh,probfield_dev,max_black_prob,min_black_prob = 127,40,0.95,0.0

def main():
    # Get the image as a grayscaled numpy array
    imagedir = "test-blurrer-2.png"
    with Image.open(imagedir) as im:
        im.show()
        imarr = np.array(ImageOps.grayscale(im))
    # Binarise the existing image to 0's (white) or 1's (black)
    imarr = binarise_image(imarr,thresh=binarise_thresh)
    # Get a matrix of distances to black based on this
    imdist = gen_distfield(imarr)
    distance_image = Image.fromarray(np.array(imdist,dtype=np.uint8))
    distance_image.show()
    # Get a matrix of probabilities that the pixel in question will be black based on this
    improb = gen_probfield(imdist,dev=probfield_dev,max_rand = max_black_prob,min_rand = min_black_prob)
    # Convert these probabilties to blacks and whites
    out = np.array(gen_binarr(improb),dtype=np.uint8)
    output_image = Image.fromarray(out)
    output_image.show()
    
# Create a binary matrix from a matrix of probabilities
def gen_binarr(probarr):
    binarr = np.zeros(probarr.shape,dtype=int)
    for i in range(len(binarr)):
        for j in range(len(binarr[i])):
            binarr[i][j] = int(np.random.choice([0,255],1,p=[probarr[i][j],1-probarr[i][j]]))
    return binarr

# Binarise an input image so pixels >127 are 1's and everything else are 0's
def binarise_image(imarr,thresh=127):
    for i in range(len(imarr)):
        for j in range(len(imarr[i])):
            if(imarr[i][j]<thresh):
                imarr[i][j] = 1
            else:
                imarr[i][j] = 0
    return imarr

# Convert a numpy array of 1's and 0's to one of probabilities based on distance to 1's and a standard deviation
def gen_probfield(imarr,dev=25.0,max_rand = 0.95,min_rand = 0.0,reweight = False):
    prob_arr = np.zeros(imarr.shape,dtype=float)
    for i in range(len(imarr)):
        for j in range(len(imarr[i])):
            prob_arr[i][j] = custom_sigmoid(imarr[i][j],dev,max_rand,min_rand)
    # If reweight is true, take the minimum and maximum probabilities, and linearly reweight them to 1 and 0
    return prob_arr

# The custom sigmoid function used to get probabilities based on distance
def custom_sigmoid(x,dev=25.0,max_rand = 0.95,min_rand = 0.0):
    return max(min(2 - np.divide(2,1+np.exp(-1*x*np.divide(1,dev*1.5))),max_rand),min_rand)

# Create a matrix of distances from black pixels
def gen_distfield(imarr,maxdist=2000,every_check=50,upperlim = np.iinfo(np.int32).max):
    distfield = np.zeros(imarr.shape,dtype=int)
    cur = []
    # Set the initial values to all infinity, while marking all the starting points
    for i in range(len(distfield)):
        for j in range(len(distfield[i])):
            distfield[i][j] = upperlim
            if(imarr[i][j]>0.5):
                cur.append([i,j].copy())
    dist = 0
    searched = {}
    while(dist<maxdist):
        # Run through the current list of positions, checking if it's a faster route to the next point
        next_layer = []
        for pos in cur:
            # If this position has been searched, continue the loop
            posid = ";".join([str(pos[0]),str(pos[1])])
            if(posid in searched):
                continue
            # Update the optimal distance to this location
            distfield[pos[0]][pos[1]] = min(distfield[pos[0]][pos[1]],dist)
            # Find the next possible search positions
            nextpos = [[pos[0]+1,pos[1]].copy(),[pos[0]-1,pos[1]].copy(),
                       [pos[0],pos[1]+1].copy(),[pos[0],pos[1]-1].copy()]
            # Remove invalid search positions
            i = 0
            while(i<len(nextpos)):
                n = nextpos[i]
                if(n[0]<0 or n[1]<0):
                    nextpos.pop(i)
                    continue
                if(n[0]>=len(distfield)):
                    nextpos.pop(i)
                    continue
                if(n[1]>=len(distfield[0])):
                    nextpos.pop(i)
                    continue
                i += 1
            next_layer += nextpos.copy()
            # Add this to the list of searched positions
            searched[";".join([str(pos[0]),str(pos[1])])] = True
        # Update the next list of positions to search through
        cur = next_layer.copy()
        dist += 1
        # If the distance can be divided by 'every_check', check if all positions have been reached and break if so
        if(dist%every_check==0):
            done = True
            for i in range(len(distfield)):
                for j in range(len(distfield[i])):
                    if(distfield[i][j]==upperlim):
                        done = False
                        break
                if(done==False):
                    break
            if(done==True):
                print(f"Finished after searching a maximum of {dist} tiles away")
                break
            print(f"Not finished after searching up to {dist} pixels away")
    return distfield
    
if(__name__=="__main__"): main()