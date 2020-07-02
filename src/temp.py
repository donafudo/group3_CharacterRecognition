import pandas as pd
s = pd.Series(list('あいうえお'))
print(s)

s=pd.Series(list('おえういあ'))
print(s)

dummie=pd.get_dummies(s)

print(dummie.columns[0])