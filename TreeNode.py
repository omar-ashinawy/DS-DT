# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
class TreeNode:
    def __init__(self, depth = 1, maxDepth = None):
        self.depth = depth
        self.maxDepth = maxDepth
        
    def train(self, X, labels):
        #first base case: checks if labels is one element or labels has one class (no possible splits) 
        if len(labels) == 1 or len(set(labels)) == 1:
            self.colmn = None
            self.split = None
            self.left = None
            self.right = None
            self.prediction = labels[0]
        else:
            numCols = range(X.shape[1])
            
            maxInfGain = 0
            bestCol = None
            bestColSplit = None
            for col in numCols:
                infGain, split = self.splitColoumn(X, labels, col)
                if infGain > maxInfGain:
                    maxInfGain = infGain
                    bestCol = col
                    bestColSplit = split
            
            if maxInfGain == 0: #no splits 
                self.colmn = None
                self.split = None
                self.left = None
                self.right = None
                self.prediction = np.round(labels.mean())
            else:
                self.colmn = bestCol
                self.split = bestColSplit
                
                if self.depth == self.maxDepth:
                    self.left = None
                    self.right = None
                    self.prediction = [
                        np.round(np.mean(labels[X[:,bestCol] < self.split])), #left node pred
                        np.round(np.mean(labels[X[:,bestCol] >= self.split])), #right node pred
                    ]
                else:
                    leftInd = (X[:, bestCol] < bestColSplit)
                    Xleft = X[leftInd]
                    labelsLeft = labels[leftInd]
                    self.left = TreeNode(self.depth + 1, self.maxDepth)
                    self.left.train(Xleft, labelsLeft) #recursion
                    
                    rightInd = (X[:, bestCol] >= bestColSplit)
                    Xright = X[rightInd]
                    labelsright = labels[rightInd]
                    self.right = TreeNode(self.depth + 1, self.maxDepth)
                    self.right.train(Xright, labelsright) #recursion

    def informationGain(self, colX, colY, split):
        yLeft = colY[colX < split]
        yRight = colY[colX >= split]
        yLen = len(colY)
        yLeftLen = len(yLeft)
        
        if (yLeftLen == 0 or yLeftLen == yLen):
            return 0
        yLeftProp = float(yLeftLen) / yLen
        yRightProp = 1 - yLeftProp
        return entropy(colY) - yLeftProp*entropy(yLeft) - yRightProp*entropy(yRight)
        
    def splitColoumn(self, X, labels, col):
        colToBeSplit_X = X[:, col]
        sortedInd = np.argsort(colToBeSplit_X)
        colToBeSplit_X = colToBeSplit_X[sortedInd]
        colToBeSplit_Y = labels[sortedInd]
        boundaries = np.nonzero(colToBeSplit_Y[:-1] != colToBeSplit_Y[1:])[0]
        
        best_split = None
        max_ig = 0
        for b in boundaries:
            split = (colToBeSplit_X[b] + colToBeSplit_X[b+1]) / 2
            ig = self.informationGain(colToBeSplit_X, colToBeSplit_Y, split)
            if ig > max_ig:
                max_ig = ig
                best_split = split
        
        return max_ig, best_split
        
    def predict_element(self, element):
        if self.colmn is not None and self.split is not None:
            if element[self.colmn] < self.split:
                if self.left:
                    p = self.left.predict_element(element)
                else:
                    p = self.prediction[0]
            else:
                if self.right:
                    p = self.right.predict_element(element)
                else:
                    p = self.prediction[1]
        else:
            p = self.prediction
        return p
    
    def predict(self, X):
        predictions = np.zeros(len(X)) #= np.zeros_like(X)
        for i in range (len(X)):
            predictions[i] = self.predict_element(X[i])
        return predictions


