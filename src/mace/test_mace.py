import pandas as pd

TEST_DATA_X = """
0,,1,,2,1,,3,,2,,3,2,1,,,0,,,
2,,,,,2,,,1,1,1,1,0,,1,,0,,,0
,2,,2,,,,,2,0,0,2,1,2,,,,,1,1
1,0,,2,1,0,,3,1,0,,,,,,,,,2,1
1,,,2,,,,2,,,0,2,1,2,,1,,,1,2
,0,2,0,,,1,0,,,0,0,0,,,2,,1,,
1,0,,,,,1,,0,0,,,1,0,,,2,,0,2
,,1,,0,2,,0,2,1,,,1,,,,2,1,2,
,,0,2,,2,,,1,1,1,,,2,,2,,1,,2
,,2,2,2,2,,,0,,,1,,1,,2,,,0,0
,,2,0,1,,,0,0,,,0,,0,0,,2,,1,
,,0,,1,,,,1,1,0,0,2,,1,1,,,2,
,2,,2,0,,,0,0,,,0,,,0,,2,,0,0
2,1,1,1,1,,,3,2,,,,0,,1,,,1,,
,0,2,0,1,,,3,,,1,,2,0,,1,2,,,
1,,0,1,0,,0,1,,0,1,1,,,,,,,2,
0,2,,,1,,1,3,2,,,,2,,1,,,,0,0
,0,,,1,,2,1,1,,,,2,,,0,,2,1,0
,1,,2,2,,1,,,1,1,,0,0,0,,,2,,
1,,,,0,2,,2,,2,2,2,,,,2,1,,,1
2,,,2,,2,1,,,0,,2,,1,,,0,2,,0
,2,,2,,1,2,,,,,,0,2,1,1,,,1,2
1,1,2,,0,,,,,0,2,1,,,,0,,2,,1
,0,2,2,2,,,1,,2,,,,,0,2,,1,2,
0,,,2,,1,,,2,,,1,0,0,,2,,2,0,
,0,,0,1,,,2,1,1,2,2,0,,1,,,,,
,0,,2,,0,,,0,,0,,,0,,2,,1,0,0
,,0,0,,,,,,2,0,1,,,0,0,2,,1,0
0,,,,1,2,2,0,,,,,2,,,0,,2,1,0
,,,,2,0,,1,,,2,,0,2,,1,1,2,,0
,,0,2,,2,1,,0,,,,,2,1,,,0,1,1
1,2,1,2,,,,,,1,,,,2,,1,1,,0,1
,,1,,,,2,0,,0,,0,,1,,2,2,2,,1
1,2,1,,,1,,,0,,2,0,,,0,2,,2,,
,,1,,,,,,2,1,0,2,1,0,,1,,0,2,
2,,1,0,1,,,0,,,,,0,2,2,,0,,,1
0,,0,,0,1,,0,2,,2,,0,,,2,,,,1
1,2,,,2,2,1,0,,,,0,,2,0,,,,2,
,,,0,1,2,0,,,0,0,2,,,1,,,,1,0
,2,,1,,,,,0,1,,,1,1,2,1,2,,2,
,2,2,,1,,,,,1,,2,1,,1,2,1,,,2
,,1,2,0,1,,3,,2,0,,,2,,,,1,2,
,,,,,,,2,1,2,2,2,,1,,1,2,,0,2
,2,0,,1,0,0,0,,0,,,2,,0,,0,,,
2,,,0,,1,,3,0,,1,,0,0,,1,2,,,
,,1,,,0,0,3,1,1,,3,,0,,1,,1,,
,,1,0,,,2,1,1,0,0,0,1,,,,,,,0
1,2,,0,,1,,,,,,0,,,0,1,0,,0,0
0,0,1,1,,0,,,,1,1,,,0,,0,,,1,
2,,2,2,,0,0,,,,2,0,1,,,,,,0,2
2,,0,,1,,0,,,1,2,2,,,,1,2,1,,
,0,1,,,0,1,,0,0,2,2,,,1,1,,,,
1,2,2,2,,,0,,,,1,0,,,,,0,,1,0
,,0,1,1,,,,0,0,,2,,1,2,,,1,0,
2,,,2,,1,,2,,1,0,,0,,0,0,,,2,
2,,1,2,,1,,,2,,0,,0,,0,0,,,,1
0,,,1,,,,3,2,2,1,3,2,,2,,,,2,
1,,,2,,2,,2,,0,1,2,,0,2,,,0,,
0,0,2,,,,2,,,,0,1,0,,,,0,,2,1
2,,2,,,0,,,0,2,1,,,,,1,2,,0,0
0,,0,0,,2,2,,0,,,,,1,0,,,1,2,
,,2,,2,2,,,1,1,,,,0,,0,,1,0,0
0,2,,2,0,,,,,0,0,,1,1,,0,,,2,
,2,,,,,,1,2,1,,,1,2,,0,,2,2,2
2,2,,,2,1,1,,0,,,0,,,0,2,,,2,
2,,2,0,2,1,1,,,,,3,,,0,1,,,1,
,0,,,0,,1,,1,,1,3,2,,1,,,,2,2
1,,0,0,,1,,2,2,,,1,,,,0,,2,0,
1,2,,0,,,1,2,,,2,2,2,,,,,2,,2
0,1,0,,,0,0,,,,,2,,,,1,,1,0,1
2,2,,,1,,,,2,2,0,0,,,2,,,2,0,
,,2,,1,,,2,,2,0,2,0,,,,1,0,1,
1,,,0,2,,1,2,2,,1,,,0,,0,,,,0
2,,1,,0,2,,,1,,,2,,,2,1,1,,,1
0,1,,,,1,,2,0,0,,,,,,0,,0,0,2
,,,2,0,,,0,,,2,,1,2,2,,0,1,2,
1,1,1,,,2,0,,,,,,1,0,1,,,2,1,
,2,1,,2,,,0,,,,0,,2,1,0,,1,,0
1,,0,,2,,,0,0,0,,,2,,2,,0,,2,
,,2,0,1,,2,2,2,0,,,,,,1,1,,1,
,,1,2,,1,,,0,1,,,2,2,1,1,,0,,
,2,,,,1,2,1,,1,,,,,0,0,1,0,0,
,,1,0,,,2,,0,,2,,0,,1,,,1,2,1
,,1,1,2,,,,0,0,,,,2,,1,,0,2,2
0,,,,2,0,1,2,2,,1,,,0,,,2,,,1
1,,2,2,,2,,3,,0,,3,1,,,,,0,2,
,0,,2,0,,1,0,,1,,,0,1,,,0,2,,
,,,0,1,0,2,1,,0,2,,,2,,,2,,2,
0,0,1,,,0,,3,0,,2,,,0,1,0,,,,
1,0,,2,2,,0,,0,,,2,0,,,1,,,0,
1,,0,1,,1,0,,1,2,,,0,,,,0,,,0
0,1,,,1,2,,,,,2,,1,,2,,2,2,,2
,2,1,,0,1,,3,,0,,3,,,2,,2,1,,
,,2,,,2,0,1,1,,1,,,,1,1,0,,0,
,1,,1,0,,,1,0,,2,1,,,0,,,,1,2
,0,0,2,2,,,,,2,2,,,,,2,1,2,2,
2,,,2,,2,1,,2,1,,,0,,2,0,,,0,
0,,,2,2,1,0,0,0,,,,1,2,,1,,,,
1,,,,0,1,0,3,,,,3,2,2,,2,1,,,
1,0,0,,,,,2,,,2,,0,1,,2,2,,,0
"""

TEST_DATA_Y = """
3
1
2
3
2
0
2
0
2
1
0
3
0
3
3
1
3
1
3
2
2
1
1
1
1
2
3
1
0
1
0
3
0
0
2
0
0
0
2
3
2
3
2
0
3
3
0
2
1
0
2
2
0
2
2
1
3
2
1
0
0
1
1
1
0
3
3
1
2
2
0
2
2
2
2
0
3
0
0
2
3
1
3
2
2
3
0
1
3
2
1
1
3
1
1
2
3
0
3
2
"""

if __name__=="__main__":
    import io
    from sklearn.metrics import accuracy_score

    X = pd.read_csv(io.StringIO(TEST_DATA_X), header=None, index_col=None).fillna(-1).values.astype(int)
    y = pd.read_csv(io.StringIO(TEST_DATA_Y), header=None, index_col=None).fillna(-1).values.astype(int)

    from mace import MACE, get_majority_vote

    m = MACE(n_iter=100, n_restarts=10, verbose=True)
    m.fit(X)
    y_hat = m.predict(X)
    print(accuracy_score(y, y_hat))

    y_hat_2 = get_majority_vote(X)
    print(accuracy_score(y, y_hat_2))
