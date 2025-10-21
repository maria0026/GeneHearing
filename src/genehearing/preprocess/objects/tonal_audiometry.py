import pandas as pd
import os
import numpy as np

class TonalAudiometry():
    def __init__(self, 
                  path, 
                  tonal_suffix,
                  columnnames,
                  air_audiometry=["AirMask", "Air"],
                  bone_audiometry=["BoneMask", "Bone"]):
        

        self.pesel_columnname = columnnames['pesel_columnname']
        self.data = pd.read_csv(path, sep=None, engine='python', dtype={self.pesel_columnname: str}, encoding='cp1252')
        print("Before dropping duplicates", self.data.shape)
        self.data.columns = self.data.columns.str.upper()
        self.data = self.data.drop_duplicates()
        print("After dropping duplicates", self.data.shape)

        self.earside_col = columnnames['audiometry_earside_columnname']
        self.date_column = columnnames['date_column']
        self.type_col = columnnames['type_column']

        self.tonal_suffix = tonal_suffix
        self.air_audiometry = air_audiometry
        self.bone_audiometry = bone_audiometry

    def filter_audiometry_type(self):
        #filter vibro
        self.data = self.data[~self.data[self.type_col].str.contains("Vibro", na=False)]
        self.data = self.data[~self.data[self.type_col].str.contains("VibroMask", na=False)]
        self.data = self.data[~self.data[self.type_col].str.contains("szumy", na=False)]
        self.data = self.data[~self.data[self.type_col].str.contains("AirErr", na=False)]
        self.data = self.data[~self.data[self.type_col].str.contains("bez aparatu, szumy", na=False)]
        self.data = self.data[~self.data[self.type_col].str.contains("UCL", na=False)]

    def patients_dfs(self):
        self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])
        self.data['date_year_month_day'] = (
            self.data[self.date_column].dt.year.astype(str) + "-" +
            self.data[self.date_column].dt.month.astype(str) + "-" +
            self.data[self.date_column].dt.day.astype(str)
        )
        self.group_columns = [self.pesel_columnname] + ['date_year_month_day'] #+ [self.type_col]

        #mini_df for each patient and each examination
        self.mini_dfs = []
        for _, group in self.data.groupby(self.group_columns):
            self.mini_dfs.append(group.reset_index(drop=True))

        print(f'Created {len(self.mini_dfs)} dataframes for each patient and each examination')

    def assign_group(self, typ):
        #return air or bone audiometry
        if typ in self.air_audiometry: 
            return "air"
        elif typ in self.bone_audiometry:
            return "bone"   


    def merge_ear(self, group, ear):
        group = group[group['ear_side']== ear] #group by ear, create a dict
        if group.shape[0] > 1:
            row_first_left = group[group['ear_side'] == ear].iloc[0]
            row_second_left = group[group['ear_side'] == ear].iloc[1]
            merged_row = row_first_left.combine_first(row_second_left) #combine masked and not masked audiometry
            #print(merged_row)
            #update values with merged from masking and not masking
            idx_first = group[group['ear_side'] == ear].index[0]
            idx_second = group[group['ear_side'] == ear].index[1]
            group.loc[idx_first] = merged_row
            #delete second row
            mini_df = group.loc[~group.index.isin([idx_second])]
            return mini_df
        else:
            return group

    def merge_mask(self):
        ears = ['L', 'P']
        for i, mini_df in enumerate(self.mini_dfs):
            mini_df["GROUP"] = mini_df[self.type_col].apply(self.assign_group)
            grouped = {g: d for g, d in mini_df.groupby("GROUP")} #group by air and bone audiometry
            all_groups_df = pd.DataFrame()
            for key in grouped:
                ear_dfs = pd.DataFrame()
                group = grouped[key]
                group['ear_side'] = group[self.earside_col].str.extract(r"(lewego|prawego)")
                group['ear_side'] = group['ear_side'].map({"lewego": "L", "prawego": "P"})
                for ear in ears:
                    ear_df = self.merge_ear(group, ear) #merge ear in each audiometry type
                    #print(ear_df)
                    ear_dfs = pd.concat([ear_dfs, ear_df], axis=0) #merge with other ear
                #print(ear_dfs)
                all_groups_df = pd.concat([all_groups_df, ear_dfs], axis=0) #merge bone and air audiometry
                #print(all_groups_df)
            self.mini_dfs[i] = all_groups_df
            #print(self.mini_dfs[i])
        print(f'Merging rows completed.')

    def compute_diff(self, mini_df, columns, suffix='_diff'):
        diff = mini_df[columns].diff().iloc[1:]  # bierzemy tylko drugi wiersz
        diff.columns = [col + suffix for col in columns]
        return diff

    def check_symmetry_def1(self, diff_df, threshold=20):
        diff_df = diff_df.dropna(axis=1, how='all')
        if diff_df.shape[1] < 2:
            return "brak_obl"
        sym = True 
        for index in range(diff_df.shape[1]-1): 
            if ((diff_df.iloc[0, index]>=threshold or diff_df.iloc[0, index]<=-threshold) & (diff_df.iloc[0, index+1]>=threshold or diff_df.iloc[0, index+1]<=-threshold)): 
                sym = False
            
        return int(sym)

    def check_symmetry_def2(self, diff_df, threshold=15):
        diff_df = diff_df.dropna(axis=1, how='all')
        if diff_df.shape[1] < 2:
            return "brak_obl"
        sym = True
        if (diff_df.iloc[0]>=threshold).sum() + (diff_df.iloc[0]<=-threshold).sum() > 1:
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


    def define_symmetry(self, first_symmetry_columns, second_symmetry_columns, suffix="_diff"):
        for i, mini_df in enumerate(self.mini_dfs):
            #mini_df["GROUP"] = mini_df[self.type_col].apply(self.assign_group)
            grouped = {g: d for g, d in mini_df.groupby("GROUP")}
            #only for air audiometry
            #if grouped.loc[0, self.type_col] in self.air_audiometry:
            for key in grouped:
                if key == "air":
                    group = grouped[key] #group inherits indexes after mini_df
                    if group.shape[0] != 2:
                        group.loc[:, 'SYMETRIA'] = "brak_obl"
                    else:
                        diff_def1 = self.compute_diff(group, first_symmetry_columns)
                        diff_def2 = self.compute_diff(group, second_symmetry_columns)

                        group['SYMETRIA_1_DEF'] = self.check_symmetry_def1(diff_def1)
                        group['SYMETRIA_2_DEF'] = self.check_symmetry_def2(diff_def2)
                        group['SYMETRIA'] = group.apply(self.combine_sym, axis=1)

                        diff_columns = [col + suffix for col in first_symmetry_columns]
                        for col in diff_columns:
                            mini_df[col] = diff_def1[col].iloc[0]

                        self.mini_dfs[i]['SYMETRIA_1_DEF'] = group['SYMETRIA_1_DEF']
                        self.mini_dfs[i]['SYMETRIA_2_DEF'] = group['SYMETRIA_2_DEF']

                    self.mini_dfs[i]['SYMETRIA'] = group['SYMETRIA']



    def calculate_pta(self, df, PTA2_columns, PTA4_columns, hf_columns):
        df['PTA2'] = df[PTA2_columns].mean(axis=1)
        df['PTA4'] = df[PTA4_columns].mean(axis=1)
        df['hfPTA'] = df[hf_columns].mean(axis=1)

    def calculate_mean_ear_pta(self, PTA2_columns, PTA4_columns, hf_columns):
        numeric_cols = PTA2_columns + PTA4_columns + hf_columns
        text_cols = [col for col in self.data.columns if col not in numeric_cols]

        for i, mini_df in enumerate(self.mini_dfs):
            grouped = {g: d for g, d in mini_df.groupby("GROUP")} #group by air and bone
            for key, group in grouped.items():
                if key == "air": #calculate only for air audiometry
                    sym = grouped["air"].iloc[0]['SYMETRIA'] #symmetry condition
                    if sym == 1 and group.shape[0] == 2:
                        df_source = pd.DataFrame({
                            **{col: [group[col].mean()] for col in numeric_cols},
                            **{col: [group[col].iloc[0]] for col in text_cols}
                        })
                    else:
                        df_source = group

                    pta2_mean, pta4_mean, ptahf_mean = self.calculate_pta(df_source, PTA2_columns, PTA4_columns, hf_columns)

                    #wstawienie wyników do group
                    if sym == 1 and group.shape[0] == 2:
                        group.loc[:, 'PTA2'] = pta2_mean.item()
                        group.loc[:, 'PTA4'] = pta4_mean.item()
                        group.loc[:, 'hfPTA'] = ptahf_mean.item()
                    else:
                        group['PTA2'] = pta2_mean
                        group['PTA4'] = pta4_mean
                        group['hfPTA'] = ptahf_mean

                
                    self.mini_dfs[i]['PTA2'] = group['PTA2']
                    self.mini_dfs[i]['PTA4'] = group['PTA4']
                    self.mini_dfs[i]['hfPTA'] = group['hfPTA']
                #result_parts.append(group)
            #print(self.mini_dfs[i])

        print('PTA calculation completed.')

    def calculate_pta(self, df, PTA2_columns, PTA4_columns, hf_columns):
        pta2_mean = df[PTA2_columns].mean(axis=1, skipna=False).round(1)
        pta4_mean = df[PTA4_columns].mean(axis=1, skipna=False).round(1)
        ptahf_mean = df[hf_columns].mean(axis=1, skipna=False).round(1)

        return pta2_mean, pta4_mean, ptahf_mean

    def save_processed_df(self, output_path):
        merged_df = pd.concat(self.mini_dfs, ignore_index=True)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        merged_df.to_csv(f'{output_path}audiometry_{self.tonal_suffix}.csv', index=False)
        print(f'Saving to {output_path}audiometry_{self.tonal_suffix}.csv completed.')


