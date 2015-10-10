#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 14/08/2014

@author: Francisco Dominguez
Easy ICP a exercise in 3D in order to learn the basic concepts
Iterative Closest Point Algorithm
In this exercise you will use a standard ICP algorithm with the point-to-point distance metric to
estimate the transform between the 2D datasets (model - red and target - green) depicted in
the below figure. For the correspondence estimation please use the nearest neighbor search
with the maximum radius set to 4 grid units. For the rotation estimation use an SVD library of
your choice or calculate it yourself. Recall the main steps in the ICP algorithm:
- find point pairs;
- compute centroids
- build the correlation matrix H;
- estimate rotation matrix R using SVD;
- estimate the translation vector t;
- calculate the transform T;
- transform dataset m using the transform T.
'''
import numpy as np
from kdtree import KDTree
from visual import vector,points

MAX_ITER      =500 #max iterations number in fitting, if reached may be divergence
MAX_DIST_MATCH=1000 #max distance to match points farther points are no considered
MIN_DIF=0.1          #min difference from model at next iteration

def distData(m1,m2):
    saDif=vector(sum(abs(m2-m1)))
    return saDif.mag

def fitICP(model,target):
    Tf=np.eye(4,4)
    dif=100
    nIter=0
    while nIter<MAX_ITER and dif>MIN_DIF:
        T1,pit1=ICPstep(model,target)
        Tf=Tf.dot(T1)
        dif=distData(model,pit1)
        #saDif=vector(sum(abs(model-pit1)))
        #dif=saDif.mag #difference with respect to the anterior model
        print nIter,dif
        points(pos=pit1,size=2,color=(1,0,1))
        model=pit1
        nIter+=1
    #print nIter,dif
    return Tf,pit1
def fitICPkdTree(model,target):
    Tf=np.eye(4,4)
    dif=100
    nIter=0
    kdTree=KDTree(list(target))
    while nIter<MAX_ITER and dif>10:
        T1,pit1=ICPstepKDTree(model,kdTree)
        Tf=Tf.dot(T1)
        saDif=vector(sum(abs(model-pit1)))
        dif=saDif.mag #difference with respect to the anterior model
        print nIter,dif
        #points(pos=pit1,color=(0,1,1))
        model=pit1
        nIter+=1
    #print nIter,dif
    return Tf,pit1
def ICPstepKDTree(pm0,kdt):
    pm,pt=findPointPairsKDtree(pm0,kdt)
    pcm,pct=computeCentroids(pm,pt)
    H=buildCorrelationMatrix(pm,pcm,pt,pct)
    R=estimateRotationMatrixUsingSVD(H)
    t=estimateTranslationVector(pcm,pct,R)
    T=calculateTransform3D(R,t)
    pmf=transformDataSetUsingTransform(pm0,T)
    return T,pmf

def ICPstep(pm0,pt0):
    pm,pt=findPointPairs(pm0,pt0)
    #pm,pt=pm0,pt0
    pcm,pct=computeCentroids(pm,pt)
    H=buildCorrelationMatrix(pm,pcm,pt,pct)
    R=estimateRotationMatrixUsingSVD(H)
    t=estimateTranslationVector(pcm,pct,R)
    T=calculateTransform3D(R,t)
    pmf=transformDataSetUsingTransform(pm0,T)
    return T,pmf
def getNearestNeighborIndex(pt,p):
    x=pt-p
    d=np.sum(x**2,axis=-1)**(0.5)
    j=np.argmin(d)#min index
    dj=d[j]       #min distance
    return j,dj
#brute force nearest neighbor
def findPointPairs(pm,pt):
    match=[]
    processedPts=set()
    for i,p in enumerate(pm):
        j,dj=getNearestNeighborIndex(pt,p)
        if dj<MAX_DIST_MATCH:
            match.append(j)
#             if not j in processedPts:
#                 match.append(j)
#                 processedPts.add(j)
#             else:
#                 match.append(-1)
        else:
            match.append(-1)    
    c=match
    pmr=[]
    ptr=[]
    for i,j in enumerate(c):
        if j!=-1:
            pmr.append(pm[i])
            ptr.append(pt[j])
    #print len(pmr)
    return np.array(pmr),np.array(ptr)
def findPointPairsKDtree(pm,kdT):
    pmr=[]
    ptr=[]
    nNeighbours=1
    processedPts=set()
    for ptm in pm:
        nnPts=kdT.query(ptm,nNeighbours)
        for ptNN in nnPts:
            if not tuple(ptNN) in processedPts:
                pmr.append(ptm)
                ptr.append(ptNN)
                processedPts.add(tuple(ptNN))
                break
    print len(pmr)
    return np.array(pmr),np.array(ptr)   
    
def computeCentroids(pm,pt):
    pcm=pm.mean(0)
    pct=pt.mean(0)
    return pcm,pct
def buildCorrelationMatrix(pm,pcm,pt,pct):
    '''
    suppose pmodel and ptarget are matched each other
    I mean pm[i] match with pt[i]
    '''
    md=pm-pcm
    td=pt-pct
    H=md.transpose().dot(td) # H=md'*td
    return H
def estimateRotationMatrixUsingSVD(H):
    [U,D,Vt]=np.linalg.svd(H) # U,D,V=SVD(H)
    V=Vt.transpose()
    R=V.dot(U.transpose()) # R=V*U'
    det=np.linalg.det(R)
    if det<0.0:
        R[:,2]=-R[:,2]
    return R
def estimateTranslationVector(pcm,pct,R):
    t=pct-R.dot(pcm.transpose()) # t=pct-R*pcm'
    return t
def calculateTransform2D(R,t):
    vR=np.vstack((R,np.array([[0,0]])))
    vt=np.array([[t[0]],[t[1]],[1]])
    T=np.hstack((vR,vt))
    return T
def calculateTransform3D(R,t):
    vR=np.vstack((R,np.array([[0,0,0]])))  # add a (0,0,0) final row or put (0,0,0) under R.
    vt=np.array([[t[0]],[t[1]],[t[2]],[1]]) 
    T=np.hstack((vR,vt)) #put vt to the right of vR
    return T
def transformDataSetUsingTransform(pm,T):
    s=np.shape(pm)
    n=s[0] # number of rows or points in pm
    #points are homogeneous colum-wise
    pmh=np.vstack((pm.transpose(),np.ones(n)))
    pt=T.dot(pmh)[:3]#unhomoegneus
    return pt.transpose()#return to points in row-wise

