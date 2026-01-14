import pandas as pd

class CSVProcessor:
    def __init__(self, path_audiometry, tonal_suffix, path_genetic, path_abr, output_path,
                match_column = "PESEL"):

        encodings_to_try = ["utf-8", "utf-8-sig", "cp1250", "latin1"]
        for enc in encodings_to_try:
            try:
                self.data_genetic = pd.read_csv(path_genetic, sep=None, engine='python', dtype={match_column: str}, encoding=enc)
                self.data_audiometry = pd.read_csv(path_audiometry, sep=None, engine='python', dtype={match_column: str}, encoding=enc)
                self.data_abr = pd.read_csv(path_abr, sep=None, engine='python', dtype={match_column: str}, encoding=enc)
                print(f"Wczytano poprawnie z encoding='{enc}'")
            except UnicodeDecodeError:
                print(f"Nieudane wczytanie przy encoding='{enc}'")
        self.data_genetic.columns = self.data_genetic.columns.str.upper()
        self.data_audiometry.columns = self.data_audiometry.columns.str.upper()
        self.data_abr.columns = self.data_abr.columns.str.upper()

        self.data_genetic[match_column] = self.data_genetic[match_column].astype(str)
        self.data_audiometry[match_column] = self.data_audiometry[match_column].astype(str)
        self.data_abr[match_column] = self.data_abr[match_column].astype(str)
        self.match_column = match_column
        self.output_path = output_path
        
        self.tonal_suffix = tonal_suffix

        
    def read_merge_genetic_audiometry(self):
        self.merged = pd.merge(self.data_genetic, self.data_audiometry, how='left', on=self.match_column)
        self.merged = self.merged.loc[:, ~self.merged.columns.str.contains('^Unnamed')]

    def merge_abr(self):
        self.merged = pd.merge(self.merged, self.data_abr, how='left', on=[self.match_column])
        print(self.merged)


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


