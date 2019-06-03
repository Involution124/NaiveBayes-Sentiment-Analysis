import math
import sys
import string 
import re
import numpy as np
import math
trainFile = "trainingSet.txt"
testFile = "testSet.txt"
outputTrain = "preprocessed_train.txt"
outputTest = "preprocessed_test.txt"

class NaiveBayes:
    """
        Naive Bayes Implementation of Sentiment Analysis 
    """
    def __init__(self, featureVectors2d,  wordVec):
        print("\t[+] Initializing Bayes Network")
        self.wordVec = wordVec
        trainData = np.array(featureVectors2d[0])
        testData = np.array(featureVectors2d[1])
        # Seperate features from classification value
        self.numFeatures = len(trainData[0])-1
        self.trainX = trainData[:, 0:self.numFeatures]
        self.trainY = np.matrix.flatten(trainData[:, self.numFeatures])
        self.numTrain = len(trainData)
        self.testX = testData[:, 0:self.numFeatures]
        self.testY = np.matrix.flatten(testData[:, self.numFeatures])
        self.numTest = len(testData)


    def Dirichlet(self, occurence, records):
        """
            Returns the uniform Dirichlet Prior for binary values
                Dirichlet Priors circumvent integer underflows, while preserving probability of max/mins for Bayes Network
        """
        return (occurence + 1)/(records + 2)

    def computePriors(self):
        """
            Compute the class priors for the naive bayes network
                Each of these values are stored into the 2d array (self.priors)
                    self.priors[x_1][0] = P(x_1 = 1 | y = 0)
                    self.priors[x_1][1] = P(x_1 = 1 | y = 1)

        """
        print("\t[+] Compute Naive Bayes class priors")
        # Priors gives the probability (x = 1 | y = 0) and (x = 1 | y = 1)
        # Note if x = 0,  (x = 0 | y = Y) = ( 1 - (x = 1 | y = Y) )
        self.priors = np.zeros((self.numFeatures, 2), dtype="float64")
        
        # Seperate data into rows where y = Y
        indices0 = np.matrix.flatten(np.argwhere(self.trainY==0))
        indices1 = np.matrix.flatten(np.argwhere(self.trainY==1))
        self.numY = len(self.trainY)
        self.numY0 = len(indices0)
        self.numY1 = len(indices1)
        # Iterate over each feature and calculate it's prior
        for col in range(self.numFeatures): 
            # Number of records where x_i = 1 and  Y = 0
            numNeg = len(np.argwhere(self.trainX[indices0, :][:, col]==1))
            # Number of records where x_1 = 1 and  Y = 1
            numPos = len(np.argwhere(self.trainX[indices1, :][:, col]==1)) 
            #print("For word " + wordVec[col] + " pos=" + str(numPos) + " neg="+str(numNeg))
            self.priors[col][0] = self.Dirichlet(numNeg, self.numY0)
            self.priors[col][1] = self.Dirichlet(numPos, self.numY1)
        
    def testTraining(self):
        self.classify(self.trainX, self.trainY)

    def testTesting(self):
        self.classify(self.testX, self.testY)

    def log(self, val):
        if(val == 0):
            return 0
        else:
            return math.log(val);

    def classify(self, xData, yData):
        """
            Computes the classification using the precomputed priors and logarithmic max
                Prints out the computed accuracy
        """
        print("\t\t[+] Classifying Data:")
        probY1 = self.numY1/self.numY
        probY0 = 1 - probY1
        numGood, numBad = 0, 0
        for ind in range(len(xData)):
            row = xData[ind]
            sum1 = 0
            sum0 = 0
            # Sum the log of each prior
            # NOTE: max(\Sigma(log(P(x|y))) == max(\Pi(log(P(x|y)))
            #   + The logarithmic sum preserves the maximum product of probabilities
            for p in range(len(row)):
                if(row[p] == 1):
                    sum1 += self.log(self.priors[p][1])
                    sum0 += self.log(self.priors[p][0])
                else:
                    sum1 += self.log(1-self.priors[p][1])
                    sum0 += self.log(1-self.priors[p][0])
            sum1+= self.log(probY1)
            sum0+= self.log(probY0)
            # If the classification matches the given value
            if((sum1>sum0 and yData[ind]) or (sum1<sum0 and not yData[ind])):
                numGood +=1
            else:
                numBad +=1
        print("\t\t[+] Accuracy = " + str(numGood/(numGood+numBad)))
            


class PreProcessor:
    """
        Textual Preprocessor to translate sentences into bag of words feature vecotrs
            InputFileNames - The file names to read data from. Sentences seperated by newlines
            outputFileNames - The file names to output the preprocessed feature vectors
    """
    def __init__(self, inputFileNames, outputFileNames):
        print("\t[+] Preprocessing data.") 
        self.featureVectors2d = []
        self.createDictionary(inputFileNames)
        for fileIndex in range(len(inputFileNames)):
            featureVectors = self.createFeatureVectors(inputFileNames[fileIndex], outputFileNames[fileIndex])
            self.featureVectors2d.append(featureVectors)

    def createDictionary(self, fileNames):
        """
            Create the word dictionary and word vector
                word vector is a list of every word read in through input file
                word dictionary is simply a dictionary of every word read from input file 
        """
        numWords = 0;
        wordDict = {}
        for fileIter in range(len(fileNames)):
            fileP = open(fileNames[fileIter])
            for line in fileP:
                line = line.translate(str.maketrans('', '', string.punctuation))
                line = line.lower()
                line = re.sub('\t', '',line)
                line = re.sub(' +', ' ',line)
                line = line.strip()
                sentence, classLabel = line.rsplit(" ", 1)
                for word in sentence.split(" "):
                    try:
                        wordDict[word] += 0 
                    except KeyError:
                        wordDict[word] = numWords
                        numWords += 1;

        self.wordDict = {}
        wordVec = []
        for key in wordDict:
            wordVec.append(key)
        wordVec.sort()
        self.wordVec = wordVec
        for ind in range(len(wordVec)):
            self.wordDict[wordVec[ind]] = ind
        self.numWords = numWords;
    
    def createFeatureVectors(self, inputFileName, outputFileName):
        """
            Create feature vectors from the input files, and output them to the ouput files
                Precondition: Word vector and dictionary must be created already
        """
        print("\t\t[+] Creating feature vector for " + inputFileName)
        outputFile = open(outputFileName, "w+")
        inputFile = open(inputFileName, "r")
        csvString = ",".join(self.wordVec) + "\n"
        outputFile.write(csvString)
        featureVectors = []
        for line in inputFile:
            featureVec = [0] * (self.numWords + 1)
            line = line.translate(str.maketrans('', '', string.punctuation))
            line = line.lower()
            line = re.sub('\t', '',line)
            line = re.sub(' +', ' ',line)
            line = line.strip()
            sentence, classLabel = line.rsplit(" ", 1)
            for word in sentence.split(" "):
                try: 
                    featureVec[self.wordDict[word]] = 1;
                except KeyError: 
                    print("Error, the word '" + word + "' was not found in dict")
            featureVec[self.numWords] = int(classLabel)
            featureVectors.append(featureVec)
            csvString = ",".join(map(str, featureVec))
            outputFile.write(csvString+"\n")
        return featureVectors


    def getBayesData(self):
        """
            Returns the feature vectors, word dictionary, and word vector
        """
        return (self.featureVectors2d, self.wordVec)



# Preprocess the data into feature vectors
Preprocessor = PreProcessor((trainFile, testFile),(outputTrain, outputTest))
# Retrieve the feature vectors and dictionary vector
featureVectors2d, wordVec = Preprocessor.getBayesData()
# Initialize the Bayes network
NBNetwork = NaiveBayes(featureVectors2d, wordVec)
# Train/Compute the priors for Naive Bayes
NBNetwork.computePriors()
# Test the data on the training set 
NBNetwork.testTraining()
# Test the data on the testing set 
NBNetwork.testTesting()
