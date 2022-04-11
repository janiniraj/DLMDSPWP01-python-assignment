import pandas as pd
from sqlalchemy import create_engine


###
# Core function accepts the csv and later converts csv dataset
# Core function toSql basically converts the data into sql
###
class CoreFunction:

    def __init__(self, csvPath):
        # csvpath param represents the input-data/filename
        # Here we are parsing the csv file into list of dataList and later iterate the object to get details
        # We need specific structure for csv in which 1st column is x and following column is y values
        self.dataFrames = []

        # The csv is being read by the Panda module and turned into a dataframe
        try:
            self.csvData = pd.read_csv(csvPath)
        except FileNotFoundError:
            print("There is an issue while reading file {}".format(csvPath))
            raise

        # x values are fetched from csvData
        xValues = self.csvData["x"]

        # I am iterating the next line for each column from panda dataframe and create new object from data
        for nameOfColumn, dataOfColumn in self.csvData.iteritems():
            if "x" in nameOfColumn:
                continue
            subset = pd.concat([xValues, dataOfColumn], axis=1)
            function = Function.fromDataframe(nameOfColumn, subset)
            self.dataFrames.append(function)

    def toSql(self, fileName, suffix):
        ###
        # This function accepts the filename and that is the name the db gets
        # the same function has suffix to specify based on original col name
        # Also, if the db file already exist in that case it will be replaced with new one
        # Here we are using sqlalchemy to handle all db related operations
        ###

        dbEngine = create_engine('sqlite:///{}.db'.format(fileName), echo=False)

        # Using dbEngine, generate and save db File
        csvDataCopied = self.csvData.copy()
        csvDataCopied.columns = [name.capitalize() + suffix for name in csvDataCopied.columns]
        csvDataCopied.set_index(csvDataCopied.columns[0], inplace=True)

        csvDataCopied.to_sql(
            fileName,
            dbEngine,
            if_exists="replace",
            index=True,
        )

    @property
    def functions(self):
        ###
        # This function returns the data frames
        ###
        return self.dataFrames

    def __iter__(self):
        ###
        # This created the function iterable
        ###
        return CoreFunctionIterator(self)

    def __repr__(self):
        return "Contains {} number of functions".format(len(self.functions))


class CoreFunctionIterator():

    def __init__(self, coreFunctionObj):
        # initialize iteration
        self.index = 0
        self.coreFunctionObject = coreFunctionObj

    def __next__(self):
        # returns next of iteration
        if self.index < len(self.coreFunctionObject.functions):
            valueRequested = self.coreFunctionObject.functions[self.index]
            self.index = self.index + 1
            return valueRequested
        raise StopIteration


class Function:

    def __init__(self, name):
        ###
        # This contains few X and Y related functions
        ###
        self._name = name
        self.dataframe = pd.DataFrame()

    def locateYBasedOnX(self, x):

        ###
        # Thus function provides Y value based on X
        # Let's say if the value is not found it throws indexError
        ###
        searchKey = self.dataframe["x"] == x
        try:
            return self.dataframe.loc[searchKey].iat[0, 1]
        except IndexError:
            raise IndexError

    @property
    def name(self):
        ###
        # A simple function to return the name
        ###
        return self._name

    def __iter__(self):
        return FunctionIterator(self)

    def __sub__(self, second):
        ###
        # This function substracts and give results
        ###
        return self.dataframe - second.dataframe

    @classmethod
    def fromDataframe(cls, name, dataframe):
        ###
        # This method is defined to get function from the data frame
        ###
        dataFunction = cls(name)
        dataFunction.dataframe = dataframe
        dataFunction.dataframe.columns = ["x", "y"]
        return dataFunction

    def __repr__(self):
        return "This is Function for {}".format(self.name)


class IdealFunction(Function):
    def __init__(self, functionData, trainingFunction, error):
        ###
        # This accepts the functionData and Training function.
        # Based on passed details, it calculated the ideal function
        # Here, the tolerance factor is necessary, which is declated below.
        ###
        super().__init__(functionData.name)
        self.dataframe = functionData.dataframe

        self.training_function = trainingFunction
        self.error = error
        self.toleranceValue = 1
        self._tolerance = 1

    def determineLargestDeviation(self, idealFunction, trainFunction):
        # Substracts two function and provide largest derivative
        distance = trainFunction - idealFunction
        distance["y"] = distance["y"].abs()
        return max(distance["y"])

    @property
    def tolerance(self):
        # returns the current tolerance value
        self._tolerance = self.toleranceFactor * self.largestDeviation
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value):
        self._tolerance = value

    @property
    def toleranceFactor(self):
        # returns the current tolerance Value
        return self.toleranceValue

    @toleranceFactor.setter
    def toleranceFactor(self, value):
        # set current tolerance value
        self.toleranceValue = value

    @property
    def largestDeviation(self):
        # Find the largest derivative and provides value accordingly
        return self.determineLargestDeviation(self, self.training_function)


class FunctionIterator:

    def __init__(self, function):
        # On iterating over a function it returns a dict that describes the point
        self._function = function
        self._index = 0

    def __next__(self):
        # On iterating over a function it returns a dict that describes the point
        if self._index < len(self._function.dataframe):
            value_requested_series = (self._function.dataframe.iloc[self._index])
            point = {"x": value_requested_series.x, "y": value_requested_series.y}
            self._index += 1
            return point
        raise StopIteration
