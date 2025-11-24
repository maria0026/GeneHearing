import pandas as pd

class CSVProcessor:
    def __init__(self, path_audiometry, tonal_suffix, path_genetic, output_path,
                match_column = "PESEL"):
        
        self.data_genetic = pd.read_csv(path_genetic, sep=None, engine='python', dtype={match_column: str}, encoding='cp1250')
        self.data_audiometry = pd.read_csv(path_audiometry, sep=None, engine='python', dtype={match_column: str}, encoding='cp1250')
        self.match_column = match_column
        self.output_path = output_path
        
        self.tonal_suffix = tonal_suffix

        
    def read_merge_genetic_audiometry(self):
        self.merged = pd.merge(self.data_audiometry, self.data_genetic, how='left', on=self.match_column)
        self.merged = self.merged.loc[:, ~self.merged.columns.str.contains('^Unnamed')]


    def translate_number_to_sex(self, number):
        if int(number) % 2 == 0:
            return 'K'
        else:
            return 'M'
        
    def decode_sex(self):
        self.merged['PESEL_PRZEDOSTATNIA'] = self.merged['PESEL'].astype(str).str[-2]
        self.merged['sex'] = self.merged['PESEL_PRZEDOSTATNIA'].apply(self.translate_number_to_sex)
        self.merged.drop('PESEL_PRZEDOSTATNIA', axis=1, inplace=True)


    def save_merged(self):
        self.merged.to_csv(f'{self.output_path}audiometry_{self.tonal_suffix}_genetic.csv', index=False, encoding='utf-8-sig')


