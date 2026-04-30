import pandas as pd

def pre_processing(raw_data):
    raw_data = pd.read_csv(raw_data) 
    
    
    return processed_data

if __name__ == "__main__":
    pre_processing("Teen_Mental_Hralth_Dataset.csv")