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
            self.merged[col] = self.merged[col].replace({'b.o.': 105})
            self.merged[col] = pd.to_numeric(self.merged[col], errors='coerce')
        print("No response values mapped successfully.")


    def replace_values(self):
        df_copy=self.merged.copy()
        for idx, row in df_copy.iterrows():
            for ear in self.ears:
                filled = False
                for col in self.optional_columns:
                    if (pd.isna(row[col + ear]) and filled):
                        row[col + ear] = 'podstawione'
                        break
                    #add ear suffix
                    column_ear = col + ear
                    if pd.isna(row[column_ear]):
                        row[column_ear] = row[self.additional_column + ear]
                        filled = True
            df_copy.loc[idx] = row
        return df_copy

    def conditional_mean(self, df, columns):
        podstawione = df[columns].eq('podstawione')
        #replace 'podstawione' with NaN and convert to numeric
        numeric = df[columns].replace('podstawione', np.nan).apply(
            pd.to_numeric, errors='coerce'
        )

        real_nans = numeric.isna() & ~podstawione #NaN które nie pochodzą z 'podstawione'

        mean = numeric.mean(axis=1, skipna=True)
        mean[real_nans.any(axis=1)] = np.nan  #te ktore mialy nan - wynik nan

        return mean.round(0)


    def calculate_PTA(self, PTA_columns):
        df_replaced = self.replace_values()
        for pta_name, columns in PTA_columns.items():
            mean_columns = []
            print(f"Calculating {pta_name} for columns: {columns}")
            for ear in self.ears:
                #add ear suffix
                column_ear = [col + ear for col in columns]
                mean_columns.extend(column_ear)
                if pta_name =='PTA4':
                    self.merged[pta_name + '_' + ear] = self.conditional_mean(df_replaced, column_ear)
                else:
                    self.merged[pta_name + '_' + ear] = self.conditional_mean(self.merged, column_ear)
            if pta_name =='PTA4':
                self.merged[pta_name + '_MEAN'] = self.conditional_mean(df_replaced, mean_columns)
            else:
                self.merged[pta_name + '_MEAN'] = self.conditional_mean(self.merged, mean_columns)


    def check_symmetry_def1(self, diff_df, threshold=20):
        diff_df = diff_df.dropna(axis=0, how='all')
        if diff_df.shape[0] < 2:
            return "brak_obl"
        sym = True
        for index in range(diff_df.shape[0]-1):
            if ((diff_df.iloc[index, 0]>=threshold or diff_df.iloc[index, 0]<=-threshold) & (diff_df.iloc[index+1, 0]>=threshold or diff_df.iloc[index+1, 0]<=-threshold)): 
                sym = False
            
        return int(sym)


    def check_symmetry_def2(self, diff_df, threshold=15):
        diff_df = diff_df.dropna(axis=0, how='all')
        if diff_df.shape[0] < 2:
            return "brak_obl"
        sym = True
        if ((diff_df >= threshold) | (diff_df <= -threshold)).sum(axis=0)[0] >= 2:
            sym = False
        return int(sym)
    

    def combine_sym(self, row):
        if row['SYMETRIA_1_DEF'] == 'brak_obl' and row['SYMETRIA_2_DEF'] == 'brak_obl':
            return 'brak_obl'
        if row['SYMETRIA_1_DEF'] == 'brak_obl':
            return int(row['SYMETRIA_2_DEF'])
        if row['SYMETRIA_2_DEF'] == 'brak_obl':
            return int(row['SYMETRIA_1_DEF'])
        else:
            return row['SYMETRIA_1_DEF'] & row['SYMETRIA_2_DEF']


    def define_symmetry(self, first_symmetry_columns, second_symmetry_columns, threshold_def1, threshold_def2, suffix="_diff"):
        self.all_columns = first_symmetry_columns
        ear_all_dict = {}
        ear_second_dict = {}

        for ear in self.ears:
            #add ear suffix
            ear_all_dict[f'ALL_COLUMNS_{ear}'] = [col + ear for col in self.all_columns]
            ear_second_dict[f'SECOND_COLUMNS_{ear}'] = [col + ear for col in second_symmetry_columns]

        for idx, row in self.merged.iterrows():
            diff_def1 = pd.DataFrame(row[ear_all_dict['ALL_COLUMNS_L']].values - row[ear_all_dict['ALL_COLUMNS_P']].values)
            row['SYMETRIA_1_DEF'] = self.check_symmetry_def1(diff_def1, threshold_def1)
            diff_def2 = pd.DataFrame(row[ear_second_dict['SECOND_COLUMNS_L']].values - row[ear_second_dict['SECOND_COLUMNS_P']].values)
            row['SYMETRIA_2_DEF'] = self.check_symmetry_def2(diff_def2, threshold_def2) 
            row['SYMETRIA'] = self.combine_sym(row) #checking conditions

            for i, col in enumerate(first_symmetry_columns):
                diff_col_name = col + suffix
                row[diff_col_name] = diff_def1.iloc[i, 0]
            self.merged.loc[idx, row.index] = row


    def select_better_ear(self, row, pta_column_L, pta_column_P):
        val_L = row[pta_column_L]
        val_P = row[pta_column_P]
        if pd.isna(val_L) or pd.isna(val_P):
            return np.nan
        if ((val_L<=20 and val_P>=35) or (val_L>=35 and val_P<=20)):
            if val_L < val_P:
                return 'niedosluch prawostronny'
            else:
                return 'niedosluch lewostronny'
        else:
            return 'brak_niesodluchu jednostronnego'
        

    def define_sidedness(self):
        for idx, row in self.merged.iterrows():
            row['SIDEDNESS'] = self.select_better_ear(row, 'PTA4_L', 'PTA4_P')
            self.merged.loc[idx, 'SIDEDNESS'] = row['SIDEDNESS']


    def map_hearing_level(self, hearing_levels, value):
        for level in hearing_levels:
            if value <= level["max"]:
                return level["label"]
            

    def classificate_hearing_loss(self, PTA_columns, biap_hearing_levels, asha_hearing_levels=None, who_hearing_levels=None):
        for pta_col in PTA_columns.keys():
            #if 'symetria' value is 1
            self.merged.loc[self.merged['SYMETRIA'] == 1, "BIAP_"+pta_col+"_MEAN"] = self.merged[pta_col+"_MEAN"].apply(lambda x: self.map_hearing_level(biap_hearing_levels, x))
            self.merged.loc[self.merged['SYMETRIA'] == 1, "ASHA_"+pta_col+"_MEAN"] = self.merged[pta_col+"_MEAN"].apply(lambda x: self.map_hearing_level(asha_hearing_levels, x))
            self.merged.loc[self.merged['SYMETRIA'] == 1, "WHO_"+pta_col+"_MEAN"] = self.merged[pta_col+"_MEAN"].apply(lambda x: self.map_hearing_level(who_hearing_levels, x))
            for ear in self.ears:
                self.merged["BIAP_"+pta_col+"_"+ear] = self.merged[pta_col+"_"+ear].apply(lambda x: self.map_hearing_level(biap_hearing_levels, x))
                self.merged["ASHA_"+pta_col+"_"+ear] = self.merged[pta_col+"_"+ear].apply(lambda x: self.map_hearing_level(asha_hearing_levels, x))
                self.merged["WHO_"+pta_col+"_"+ear] = self.merged[pta_col+"_"+ear].apply(lambda x: self.map_hearing_level(who_hearing_levels, x))


    def save_to_csv(self):
        self.merged.to_csv(f'{self.output_path}', index=False, sep=";", encoding="utf-8-sig")
        print("PTA values calculated successfully.")