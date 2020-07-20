import pandas as pd

def sort_pt(file):
    df = pd.read_csv(file, index_col=0)
    df = df.sort_values(by = 'Name')
    df.to_csv(file)

def mean_pd(x,y):
    assert x.shape == y.shape
    assert all (x.columns == y.columns)
    
    x = x.sort_values(by = 'Name')
    y = y.sort_values(by = 'Name')
    
    return (x + y) / 2

def avg_pt(name, file1, file2, file3 = None):
    df1 = pd.read_csv(file1, index_col=0)
    df2 = pd.read_csv(file2, index_col=0)
    result = mean_pd(df1, df2)
    
    if file3 != None:
        df3 = pd.read_csv(file3, index_col=0)
        result = mean_pd(result, df3) 
    
    result.to_csv(name)
    return True

def group_tf(file):
    df = pd.read_csv(file)

    df = df[['Type', 'Operation', '#Occurrences', 'Avg. self-time (us)']]
    
    sorted_ = df.sort_values(by = 'Type', ascending = True)
    grouped = df[['Type', '#Occurrences', 'Avg. self-time (us)']].groupby('Type', as_index = False).sum()
    
    return sorted_, grouped

def keep_keys(diction_ls, keys_to_keep):
    return [{ key: diction[key] for key in keys_to_keep } for diction in diction_ls]