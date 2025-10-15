import pandas as pd
import os

class ExcelProcessor:
    def __init__(self, path, audiometry_names=['Audiometria_Tonalna', 'Audiometria_Slowna', 'Audiometria_Pole_Swobodne'], 
                audiometry_map = {'Audiometria_Tonalna': 'tonal',
                                   'Audiometria_Slowna': 'verbal',
                                   'Audiometria_Pole_Swobodne': 'free_field'},
                audiometry_dropcolumns = ["WYNIKI_POZOSTALE", "WYKONUJACY_BADANIE", "OPIS_BADANIA", "id"],
                match_columns = ["PESEL", "NUMER_W_JEDNOSTCE", "NUMER_HISTORII_CHOROBY"]):
        
        self.f = pd.ExcelFile(path, engine='openpyxl')
        self.audiometry_names = audiometry_names
        self.audiometry_map = audiometry_map
        self.audiometry_dropcolumns = audiometry_dropcolumns
        self.match_columns = match_columns
        
        self.audiometries = {}

        
    def read_audiometry(self, pesel_column):
        exclude_cols = self.match_columns + self.audiometry_dropcolumns
        for sheet in self.f.sheet_names:
            for audiometry_type in self.audiometry_names:
                if sheet == audiometry_type:
                    try:
                        df = self.f.parse(sheet, dtype={pesel_column: str})
                        #give different names for all columns except the matching ones and columns to drop
                        df = df.rename(columns={col: f"{col}_{self.audiometry_map[sheet]}" for col in df.columns if col not in exclude_cols})
                        self.audiometries[self.audiometry_map[sheet]] = df
                        print(f'{self.audiometry_map[sheet]} audiometry reading completed')
                    except ValueError as e:
                        print(f"Problem with reading '{sheet}': {e}")


    def read_patients(self, patients_sheetname, pesel_column):
        for sheet in self.f.sheet_names:
            if sheet == patients_sheetname:
                self.df_patients = self.f.parse(sheet, dtype={pesel_column: str})
        print('Patient sheet reading completed.')        


    def filter_audiometry(self, description_columnname):
        for key, df in self.audiometries.items():
            df = df[df[description_columnname].isnull()]
            df = df.drop(
                columns=self.audiometry_dropcolumns
            )
            self.audiometries[key] = df
        print('Filtering audiometry completed.')
            
    def merge_audiometries(self, audiometry_type_columnname, output_path):
        
        for key, df in self.audiometries.items():
            df_merged = pd.merge(
            self.df_patients,
            df,
            on=self.match_columns,
            how="left"
            )
            col_name = f'{audiometry_type_columnname}_{key}'

            if col_name in df_merged.columns:
                df_merged = df_merged[~df_merged[col_name].isnull()]
                print(f'Dataframe for {key} audiometry merged and ready to save.')
            else:
                print(f"Warning: Column '{col_name}' not found in merged DataFrame for key '{key}'. No filtering applied.")

            self.audiometries[key] = df_merged
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            df_merged.to_csv(f'{output_path}audiometry_{key}.csv')
            print(f'Saving to {output_path}audiometry_{key}.csv completed.')

