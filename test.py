from pandas import DataFrame
from numpy import arange

df = DataFrame(arange(36).reshape(12, 3), columns=['f', 's', 't'], index=[1, 3, 4, 8, 9, 10, 12, 13, 14, 15, 18, 19])

print(df.loc[5:11])
1516671540 - 1512543180