from sklearn.ensemble import IsolationForest
from data_set import clickhouse_dataset
from scipy import stats
import pandas as pd
import numpy as np

df = pd.DataFrame(clickhouse_dataset(predictions_number=12,
                                     interval=20)[0])
clf = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.1, max_features=1.0,
                      bootstrap=False, n_jobs=1, random_state=None, verbose=0)
X = np.array(df[[2]])
r = clf.fit(X)

a_prob = clf.decision_function(X)
threshold = stats.scoreatpercentile(a_prob, 100 * 0.1)

print (a_prob)
