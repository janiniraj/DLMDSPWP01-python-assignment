from sqlalchemy import create_engine, Table, Column, String, Float, MetaData

def writeToSqlite(data):
    ###
    # Here, we are using default engine of sqline and creating tables in the sqlife for the mapping part
    ###
    dbEngine = create_engine('sqlite:///{}.db'.format("mapping"), echo=False)
    metadata = MetaData(dbEngine)

    mappingTableSchema = Table('testMappingData', metadata,
                    Column('X (test function)', Float, primary_key=False),
                    Column('Y (test function)', Float),
                    Column('Delta Y (test function)', Float),
                    Column('Number of ideal function', String(50))
                    )

    metadata.create_all()

    tableData = []
    for singleRaw in data:
        point = singleRaw["point"]
        classification = singleRaw["classification"]
        yDelta = singleRaw["delta_y"]

        if classification is not None:
            classificationName = classification.name.replace("y", "N")
        else:
            # If there is no classification, there is also no distance. In that case I write a dash
            classificationName = "-"
            yDelta = -1

        # Make sure the column name should be same as table schema, otherwise it will thorw an error
        tableData.append(
            {"X (test function)": point["x"], "Y (test function)": point["y"], "Delta Y (test function)": yDelta,
             "Number of ideal function": classificationName})

    # here we are inserting data and executing the query
    query = mappingTableSchema.insert()
    query.execute(tableData)
