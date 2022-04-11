from bokeh.plotting import figure, output_file, show
from bokeh.layouts import column, grid
from bokeh.models import Band, ColumnDataSource


def plotIdealFunctions(idealFunctions, fileName):
    # this function converts the ideal functions into graph and adds them into html file
    idealFunctions.sort(key=lambda ideal_function: ideal_function.training_function.name, reverse=False)
    graphPlots = []
    for idealFunction in idealFunctions:
        graphData = createGraphFromTwoFunctions(lineFunction=idealFunction, scatterFunction=idealFunction.training_function,
                                                squaredError=idealFunction.error)
        graphPlots.append(graphData)
    output_file("{}.html".format(fileName))
    # Shows the generated graph with file
    show(column(*graphPlots))


def createPlottingPointBasedOnIdealFunction(classificationPoints, fileName):
    # this function creates plotting points based on Ideal functions and saves in html file
    graphPlots = []
    for index, item in enumerate(classificationPoints):
        if item["classification"] is not None:
            p = classificationGraphPlot(item["point"], item["classification"])
            graphPlots.append(p)
    output_file("{}.html".format(fileName))
    show(column(*graphPlots))


def createGraphFromTwoFunctions(scatterFunction, lineFunction, squaredError):
    # This function creates graph based on two functions

    # first function dataframes and names
    functionOneDataframe = scatterFunction.dataframe
    functionOneName = scatterFunction.name

    # Second function dataframes and names
    functionTwoDataframe = lineFunction.dataframe
    functionTwoName = lineFunction.name

    # get squared error rounded to two
    squaredError = round(squaredError, 2)

    graphPlot = figure(title="Graph for train model {} vs ideal {}. Calculated Squared error = {}".format(functionOneName, functionTwoName, squaredError),
               x_axis_label='x', y_axis_label='y')
    graphPlot.scatter(functionOneDataframe["x"], functionOneDataframe["y"], fill_color="green", legend_label="Train")
    graphPlot.line(functionTwoDataframe["x"], functionTwoDataframe["y"], legend_label="Ideal", line_width=5)
    return graphPlot


def classificationGraphPlot(point, idealFunction):
    # This function plot the classification based on points.

    if idealFunction is not None:
        functionClassificationDataframe = idealFunction.dataframe
        # Get string points of Y based on X
        pointString = "({},{})".format(point["x"], round(point["y"], 2))
        title = "point: {} with classification: {}".format(pointString, idealFunction.name)

        graphPlot = figure(title=title, x_axis_label='x', y_axis_label='y')

        # draw lines for the provided ideal function
        graphPlot.line(functionClassificationDataframe["x"], functionClassificationDataframe["y"],
                legend_label="Classification function", line_width=2, line_color='black')

        # This one shows the tolerance for the ideal function in the graph
        idealFunctionTolerance = idealFunction.tolerance
        functionClassificationDataframe['upper'] = functionClassificationDataframe['y'] + idealFunctionTolerance
        functionClassificationDataframe['lower'] = functionClassificationDataframe['y'] - idealFunctionTolerance

        dataSrc = ColumnDataSource(functionClassificationDataframe.reset_index())

        band = Band(base='x', lower='lower', upper='upper', source=dataSrc, level='underlay',
            fill_alpha=0.5, line_width=4, line_color='red', fill_color="red")

        graphPlot.add_layout(band)
        graphPlot.scatter([point["x"]], [round(point["y"], 4)], fill_color="red", legend_label="Test CSV points", size=8)

        return graphPlot
