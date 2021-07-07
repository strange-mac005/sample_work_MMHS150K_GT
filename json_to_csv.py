import json
import pandas as pd
import numpy as np
import csv
with open('MMHS150K_GT.json') as f:
  data = json.load(f)
print(type(data))
df=pd.DataFrame(data.values())
df=df.to_numpy()
ke=data.keys()
ke=pd.DataFrame(ke)
ke=ke.to_numpy()
ke=np.array(ke,dtype='S')
print(ke.dtype)
final=np.append(df, ke, axis=1)
DF = pd.DataFrame(final,columns=['img_url', 'labels','tweet_url','tweet_text','labels-str','tweetID'])
DF.to_csv("output.csv")
df=pd.read_csv("output.csv")
print(df['tweetID'].astype('object'))