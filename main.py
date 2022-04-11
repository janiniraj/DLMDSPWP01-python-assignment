from functions import CoreFunction
from mappingTable import writeToSqlite
from regression import minimiseLoss, findClassification, errorSquared
from plotting import plotIdealFunctions, createPlottingPointBasedOnIdealFunction
import math
import pandas as pd
###
# This feature is developed for below functionality
# 1. Convert csv data to SQLlite
# 2. Find ideal function based on data
# 3. Visualize logical graphs
# Also, I have added few unit tests based on features.
###

if __name__ == '__main__':
    # Input Data declaration
    idealCsvPath = "input-data/ideal.csv"
    trainCsvPath = "input-data/train.csv"
    testCsvPath = "input-data/test.csv"

    # Core Function accepts the csv path, reads the file and convert to dataset
    # Later the same class has function to convert to sql which basically converts the uploaded csv file data into sql.
    idealCsvDataset = CoreFunction(csvPath=idealCsvPath)
    trainCsvDataset = CoreFunction(csvPath=trainCsvPath)
    testCsvDataset = CoreFunction(csvPath=testCsvPath)

    # convert above 2 files to sql using panda
    idealCsvDataset.toSql(fileName="ideal", suffix=" (ideal function)")
    trainCsvDataset.toSql(fileName="training", suffix=" (training function)")

    # Here we compute the ideal functions for the provided data
    # We are storing all ideal functions in the list
    idealFunctions = []
    for trainFunction in trainCsvDataset:
        # find best fitting function
        idealFunction = minimiseLoss(trainFunction=trainFunction,
                                     listOfCandidateFunctions=idealCsvDataset.functions,
                                     lossFunction=errorSquared)
        # set the tolerance factor
        idealFunction.toleranceFactor = math.sqrt(2)
        # Add the ideal function in the list
        idealFunctions.append(idealFunction)

    # Let's plot the ideal function the graph plot and save to the html file
    plotIdealFunctions(idealFunctions, "plotIdealFunction")

    # Let's fetch test CSV datasets and plot
    testDataSetPoints = testCsvDataset.functions[0]

    testDataSetIdealFunctionPoints = []
    for point in testDataSetPoints:
        idealFunction, yDelta = findClassification(point=point, idealFunctions=idealFunctions)
        result = {"point": point, "classification": idealFunction, "delta_y": yDelta}
        testDataSetIdealFunctionPoints.append(result)

    # let's plot the test data into the bokeh graph
    createPlottingPointBasedOnIdealFunction(testDataSetIdealFunctionPoints, "plottingTestDataSetIdealFunction")
    # Write the mapping to the sqlite to export as .db file
    writeToSqlite(testDataSetIdealFunctionPoints)
    print("The above code created below files",
          "\n training.csv file data as sqlite db file: training.db",
          "\n ideal.csv file data as sqlite db file: ideal.db",
          "\n mapping of point test based on test.csv for the ideal function and it's y-delta computation.",
          "\n best fitting ideal function graph and scattered data as graph: plotIdealFunction.html",
          "\n matching ideal function and distance between both functions.: plottingTestDataSetIdealFunction.html",
          "\n Created By: Niraj Jani",
          "\n Matriculation: 32202983"
          "\n Code works fine.")


