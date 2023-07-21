import pandas as pd
import re

def int_match(s):
    match=re.search("\d+",s)
    result=int(match.group(0))
    return result

def main():
    # By default, pandas inserts NaN values into empty cells
    data=pd.read_csv('paintings_all_data_info.csv')
    # Drop rows with missing date entries
    data=data.dropna(axis=0, subset=['date'])
    data['date']=data.date.apply(lambda x:int_match(str(x)))
    data.to_csv('filtered_paintings_all_data_info.csv')

if __name__ == '__main__':
    main()
