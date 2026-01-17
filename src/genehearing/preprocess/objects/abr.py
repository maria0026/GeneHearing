import pandas as pd
import os
import numpy as np 
import operator

class AuditoryBrainstemResponse():
    def __init__(self, path_abr, path_genetic, columnnames, output_path, match_column = "PESEL"):

        encodings_to_try = ["utf-8", "utf-8-sig", "cp1250"]
        for enc in encodings_to_try:
            try:
                self.data_abr = pd.read_csv(path_abr, sep=None, engine='python', dtype={match_column: str}, encoding=enc)
                self.data_genetic = pd.read_csv(path_genetic, sep=None, engine='python', dtype={match_column: str}, encoding=enc)
                print(f"Wczytano poprawnie z encoding='{enc}'")
            except UnicodeDecodeError:
                print(f"Nieudane wczytanie przy encoding='{enc}'")
        self.data_abr.columns = self.data_abr.columns.str.upper()

        self.data_abr[match_column] = self.data_abr[match_column].astype(str)
        self.match_column = match_column
        self.output_path = output_path
        self.ears = ['L', 'P']

        self.columns_to_fill = columnnames["columns_to_fill"]
        self.optional_columns = columnnames["optional_columns"]
        self.additional_column = columnnames["additional_column"]

    def merge_genetic_abr(self):
        self.merged = pd.merge(self.data_abr, self.data_genetic, how='left', on=[self.match_column])
        print(self.merged)
        print("ABR and genetic data merged successfully.")


    def map_no_response(self):
        for col in self.columns_to_fill:
            self.merged[col] = self.merged[col].replace({'b.o.': 120})
            self.merged[col] = pd.to_numeric(self.merged[col], errors='coerce')
        print("No response values mapped successfully.")


    def replace_values(self):
        for idx, row in self.merged.iterrows():
            for col in self.optional_columns:
                for ear in self.ears:
                    #add ear suffix
                    column_ear = col + ear
                    if pd.isna(row[column_ear]):
                        row[column_ear] = row[self.additional_column + ear]
            self.merged.loc[idx] = row


    def calculate_PTA(self, PTA_columns):
        for pta_name, columns in PTA_columns.items():
            mean_columns = []
            print(f"Calculating {pta_name} for columns: {columns}")
            for ear in self.ears:
                #add ear suffix
                column_ear = [col + ear for col in columns]
                mean_columns.extend(column_ear)
                self.merged[pta_name + '_' + ear] = self.merged[column_ear].mean(axis=1)
            self.merged[pta_name + '_mean'] = self.merged[mean_columns].mean(axis=1)
        self.merged.to_csv(f'{self.output_path}', index=False, sep=";", encoding="utf-8-sig")
        print("PTA values calculated successfully.")

