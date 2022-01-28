import os
import sys
import numpy as np
import pandas as pd
import torch

from pprint import pprint, PrettyPrinter
pp = PrettyPrinter()

from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession.builder\
        .master('local')\
        .appName('tensor_basic').getOrCreate()

    t = torch.randint(0, 10, (3, 5))
    t1 = t.T

    df = pd.DataFrame(t.numpy())

    if 0:
        pp.pprint({
            't': t,
            't1': t1,
            'df': df
        })

    spark_df = spark.createDataFrame(data=df)
    spark_df.printSchema()

    spark_df.show()

    # spark_df.write.csv(sys.argv[1])

