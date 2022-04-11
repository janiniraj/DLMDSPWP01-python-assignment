from functions import IdealFunction

def minimiseLoss(trainFunction, listOfCandidateFunctions, lossFunction):
    ###
    # This function find the min loss based on train function and candidate functions
    # Initially the smalled error is non and it stored the error from the for loop
    ###
    functionWithSmallestError = None
    smallestError = None
    for candidateFunction in listOfCandidateFunctions:
        error = lossFunction(trainFunction, candidateFunction)
        if ((smallestError == None) or error < smallestError):
            smallestError = error
            functionWithSmallestError = candidateFunction

    return IdealFunction(functionData=functionWithSmallestError, trainingFunction=trainFunction,
                                   error=smallestError)


def findClassification(point, idealFunctions):
    ###
    # This function find the classification of points respective to the ideal function
    ###
    lowestClassfication = None
    lowestDistance = None

    for idealFunction in idealFunctions:
        try:
            yLocation = idealFunction.locateYBasedOnX(point["x"])
        except IndexError:
            print("There is an index error for the point.")
            raise IndexError

        # finds the absolute absoluteDistance
        absoluteDistance = abs(yLocation - point["y"])

        if (abs(absoluteDistance < idealFunction.tolerance)):
            # returns the lowest distance
            if ((lowestClassfication == None) or (absoluteDistance < lowestDistance)):
                lowestClassfication = idealFunction
                lowestDistance = absoluteDistance

    return lowestClassfication, lowestDistance

def errorSquared(firstFunction, secondFunction):
    # Calculates Squared error based on two function's distance
    distances = secondFunction - firstFunction
    distances["y"] = distances["y"] ** 2
    return sum(distances["y"])
