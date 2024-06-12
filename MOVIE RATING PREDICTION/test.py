import  pandas as pd

data=pd.read_csv('movie_dataset.csv',encoding='latin1')

s_data=data.sample(n=5)

print(s_data)