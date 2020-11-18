import sys, os
import json 
from scipy.signal import medfilt

import numpy as np

def getChangePoints(power, likelihoods, windowSize=50, minDist=50):
    changeIndices = []
    win = int(windowSize/2)
    absolutLikelihood = np.abs(likelihoods)
    newPower = np.array(power)

    nonZero = np.where(absolutLikelihood > 0)[0]
    checkIndex = 0
    newGroups = []
    if len(nonZero) > 0:
        groupednonZero = np.split(nonZero, np.where(np.diff(nonZero) != 1)[0]+1)
        newGroups = groupednonZero

        for group in newGroups:
            if len(group) > 2:
                if checkIndex > group[0]: 
                    continue
                myGroup = group
                
                oldMean = newPower[group[0]-1]
                mean = newPower[group[-1]+1]
                #i = np.argmax(absolutLikelihood[myGroup])+group[0]
                if abs(mean - oldMean) < max(oldMean, mean)*0.1:
                    i = np.argmax(absolutLikelihood[myGroup])+group[0]
                if mean > oldMean:
                    i = np.argmax(likelihoods[myGroup])+group[0]
                else:
                    i = np.argmin(likelihoods[myGroup])+group[0]
                checkIndex = i + minDist
                changeIndices.append(i)

    return changeIndices

def rolling_window(a, window):
    pad = np.ones(len(a.shape), dtype=np.int32)
    pad[-1] = window-1
    pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
    a = np.pad(a, pad,mode='reflect')
        
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def pereiraLikelihood(theData, threshold=5.0, preEventLength=150, postEventLength=100, verbose=False, linearFactor=0.005):

    data = medfilt(theData, kernel_size=9)
    eventLikelihoods = np.zeros(len(data))
    means_0 = np.mean(rolling_window(data, preEventLength), axis=-1)
    means_1 = np.mean(rolling_window(data, postEventLength), axis=-1)
    std_0 = np.std(rolling_window(data, preEventLength), axis=-1)
    std_1 = np.std(rolling_window(data, postEventLength), axis=-1)

    std_0[np.where(std_0 < 0.01)] = 0.01
    std_1[np.where(std_1 < 0.01)] = 0.01        # Those values don't happen in reality, so increase them
         
    leng = len(data)
    for i in range(preEventLength, leng - postEventLength):
        j = i+postEventLength
        thres = threshold + means_0[i]*linearFactor
        if abs(means_1[j] - means_0[i]) < thres or std_1[j] == 0 or std_0[i] == 0:
            pass
            #likelihood = 0
        else:
            likelihood = np.log(std_0[i]/std_1[j]) + (data[i] - means_0[i])**2/(2*std_0[i]**2) - (data[i] - means_1[j])**2/(2*std_1[j]**2)
            eventLikelihoods[i] = likelihood

    return eventLikelihoods


def findEvents(power, thres, pre, post, voting, minDist, m):
    likelihoods = pereiraLikelihood(power, threshold=thres, preEventLength=pre, postEventLength=post, linearFactor=m, verbose=True)
    # Get change indices
    changeIndices = getChangePoints(power, likelihoods, windowSize=voting, minDist=minDist)

    return changeIndices

def findUniqueStates(power, changeIndices, thres, minDist, allowSelfLoops=True):
    LINE_NOISE = 1.0

    # Get State Seuence from all state changes
    # Handle start state
    stateSequence = [{'index': 0, 'endIndex': changeIndices[0] if len(changeIndices) > 0 else len(power)}]
    # Changes in between
    for i, change in enumerate(changeIndices[:-1]):
        stateSequence.append({'index': change, 'endIndex': changeIndices[i+1]})
    # handle end state
    if len(changeIndices) > 0: stateSequence.append({'index': changeIndices[-1], 'endIndex': len(power)-1})


    # Get Steady states point after each state change
    for i in range(len(stateSequence)):
        slice = power[ stateSequence[i]['index'] : stateSequence[i]['endIndex'] ]
        stateSequence[i]['ssIndex'] = int(stateSequence[i]['index']+minDist )
        stateSequence[i]['ssEndIndex'] = int(max(stateSequence[i]['endIndex']-minDist/2, stateSequence[i]['ssIndex']+1))
        
    # Construct mean value of state
    for i in range(len(stateSequence)):
        if stateSequence[i]['ssIndex'] is None or stateSequence[i]['ssEndIndex'] is None or stateSequence[i]['ssEndIndex'] - stateSequence[i]['ssIndex'] < 1:
            stateSequence[i]['mean'] = None
        else:
            stateSequence[i]['mean'] = np.mean(power[stateSequence[i]['ssIndex']:stateSequence[i]['ssEndIndex']])
            if stateSequence[i]['mean'] <= LINE_NOISE: stateSequence[i]['mean'] = 0


    means = sorted([stateSequence[i]['mean'] for i in range(len(stateSequence))])
    cluster = 0
    clusters = [0]
    
    for i in range(1, len(means)):
        if abs(means[i-1]-means[i]) > thres:
            cluster += 1
        clusters.append(cluster)

    for i in range(len(stateSequence)):
        stateSequence[i]["stateID"] = clusters[means.index(stateSequence[i]['mean'])]

    # prevent Self loops
    if allowSelfLoops == False:
        if len(stateSequence) > 1:
            newStateSequence = []
            source = stateSequence[0]
            for i in range(len(stateSequence)-1):
                dest = stateSequence[i+1]
                if source["stateID"] == dest["stateID"]:
                    source['endIndex'] = dest["endIndex"]
                    source['ssEndIndex'] = dest["ssEndIndex"]
                    #recalculate mean based on the length of the arrays
                    source['mean'] = (source['mean'] * (source['endIndex'] - source['index']) + dest["mean"] * (dest['endIndex'] - dest['index']))/(dest['endIndex'] - source['index'])
                else:
                    newStateSequence.append(source)
                    if dest == stateSequence[-1]:
                        newStateSequence.append(dest)
                    source = dest
            stateSequence = newStateSequence

    return stateSequence


def autoLabel(power:np.array, sr:float, thres:float=5.0, preEventTime:float=1.0, postEventTime:float=1.0, votingTime:float=2.0, minDistance:float=1.0, m:float=0.005, verbose:bool=True ):

    pre = max(int(preEventTime*sr), 1)
    post = max(int(postEventTime*sr), 1)
    voting = max(int(votingTime*sr), 1)
    
    minDist = max(minDistance*sr, 1)
    
    if verbose:
        print("sr: {}Hz, thres: {}W, pre: {}samples, post: {}:samples, voting: {}samples, minDist: {} samples, m:{}".format(sr, thres, pre, post, voting, minDist, m), flush=True)   
        print("Finding Events...")
    changeIndices = findEvents(power, thres, pre, post, voting, minDist, m)

    if verbose: print("Clustering Events...")
    
    stateSequence = findUniqueStates(power, changeIndices, thres, minDist)

    if len(changeIndices) == 0:
        if verbose: print("No Changes found in signal...")

    if len(changeIndices) >= 200:
        if verbose: print("Too many events found, you may want to change settings")

    if len(changeIndices) == 0:
        if verbose: print("Generating Labels...")
    
    
    eventIndices = [i["index"] for i in stateSequence]
    states = ["S" + str(i["stateID"]) for i in stateSequence]

    return eventIndices, states
