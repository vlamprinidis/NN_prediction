import pandas as pd
import pickle
    
def save(data, fname='data.pkl'):
    with open(fname, 'wb') as fp:
        pickle.dump(data, fp)
    fp.close()
    
def load(fname='data.pkl'):
    with open(fname, 'rb') as fp:
        data = pickle.load(fp)
    fp.close()
        
    return data

def insert_df(file,name,batch,nodes,df):
    data = load(file)
    data[name, batch, nodes] = df
    save(data, file)
       
def diff_tf(df,pred):
    s = pd.merge(df, pred, how='inner', on=['Type','Operation'])
    s['Error'] = abs(s['Avg. self-time (us)_x'] - s['Avg. self-time (us)_y'])
    s['Error %'] = 100*abs(s['Avg. self-time (us)_x'] - s['Avg. self-time (us)_y'])/s['Avg. self-time (us)_x']    
    
    return s

def from_tf(nops):
    nops = pd.DataFrame.from_dict(nops['node'])
    nops = nops[['name','op']].sort_values(by = ['op','name']).reset_index(drop=True)
    
    return nops