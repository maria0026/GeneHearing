import pandas as pd

class CSVProcessor:
    def __init__(self, path_audiometry, tonal_suffix, path_genetic, output_path,
                match_column = "PESEL"):
        
        self.data_genetic = pd.read_csv(path_genetic, sep=None, engine='python', dtype={match_column: str}, encoding='cp1252')
        self.data_audiometry = pd.read_csv(path_audiometry, sep=None, engine='python', dtype={match_column: str}, encoding='cp1252')
        self.match_column = match_column
        self.output_path = output_path
        
        self.tonal_suffix = tonal_suffix

        
    def read_merge_genetic_audiometry(self):
        self.merged = pd.merge(self.data_audiometry, self.data_genetic, how='left', on=self.match_column)


    def save_merged(self):
        self.merged.to_csv(f'{self.output_path}audiometry_{self.tonal_suffix}_genetic.csv', index=False)


