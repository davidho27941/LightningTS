import os
import pandas as pd

from glob import glob

from .base import BaseDataIO

class CSVDataIO(BaseDataIO):
    def __init__(self, file_path, **kwargs):
        super().__init__(self, file_path)
        
        self.file_path = file_path
        self.kwargs = kwargs
        
    def read(self):
        
        if os.path.isdir:
            files = sorted(glob(f"{self.file_path}/*.csv"))
            self.data = pd.DataFrame()
            for file in files:
                self.data = pd.concat(
                    [self.data, pd.read_csv(file, self.kwargs)],
                    ignore_index=True,
                )
            
        else:
            self.data = pd.read_csv(self.file_path, self.kwargs)
       
    @property 
    def data(self):
        return self.data
    
class ParquetDataIO(BaseDataIO):
    def __init__(self, file_path, **kwargs):
        super().__init__(self, file_path)
        
        self.file_path = file_path
        self.kwargs = kwargs
        
    def read(self):
        
        if os.path.isdir:
            files = sorted(glob(f"{self.file_path}/*.parquet"))
            self.data = pd.DataFrame()
            for file in files:
                self.data = pd.concat(
                    [self.data, pd.read_parquet(file, self.kwargs)],
                    ignore_index=True,
                )
            
        else:
            self.data = pd.read_parquet(self.file_path, self.kwargs)
       
    @property 
    def data(self):
        return self.data