import pandas as pd
import os
import numpy as np 
import operator

class TonalAudiometry():
    def __init__(self, 
                  path, 
                  tonal_suffix,
                  implants_datapath,
                  columnnames,
                  implant_columnnames,
                  air_audiometry=["AirMask", "Air"],
                  bone_audiometry=["BoneMask", "Bone"],
                  vibro_audiometry=["Vibro", "VibroMask"]):
        

        self.patient_number_columnname = columnnames['patient_number_columnname']
        self.data = pd.read_csv(path, sep=None, engine='python', dtype={self.patient_number_columnname: str}, encoding='utf-8-sig')
        print("Before dropping duplicates", self.data.shape)
        self.data.columns = self.data.columns.str.upper()
        self.data = self.data.drop_duplicates()
        print("After dropping duplicates", self.data.shape)

        self.earside_col = columnnames['audiometry_earside_columnname']
        self.date_column = columnnames['date_column']
        self.type_col = columnnames['type_column']
        self.description_col = columnnames['description_column']

        self.tonal_suffix = tonal_suffix
        self.air_audiometry = air_audiometry
        self.bone_audiometry = bone_audiometry
        self.vibro_audiometry = vibro_audiometry
        self.ears = ['L', 'P']

        self.path_implants = implants_datapath
        self.genetic_patient_id_column = columnnames['genetic_patient_id_column']

        self.implant_ear_columnname = implant_columnnames['implant_ear_columnname']
        self.implant_date_columnname = implant_columnnames['implant_date_columnname']
        self.implant_ear_second_columnname = implant_columnnames['second_implant_ear_columnname']
        self.implant_date_second_columnname = implant_columnnames['second_implant_date_columnname']


    def merge_implants(self):
        implanty = pd.read_csv(self.path_implants, sep=None, engine='python', dtype={self.genetic_patient_id_column: str}, encoding='cp1250')
        implanty.columns = implanty.columns.str.upper()
        
        self.data = pd.merge(self.data, implanty, how='left', on=self.genetic_patient_id_column)


    def filter_audiometry_type(self):
        #filter vibro
        #self.data = self.data[~self.data[self.type_col].str.contains("Vibro", na=False)]
        #self.data = self.data[~self.data[self.type_col].str.contains("VibroMask", na=False)]
        self.data = self.data[self.data[self.description_col].isna() | (self.data[self.description_col] == "")]
        self.data = self.data[~self.data[self.type_col].str.contains("szumy", na=False)]
        self.data = self.data[~self.data[self.type_col].str.contains("AirErr", na=False)]
        self.data = self.data[~self.data[self.type_col].str.contains("bez aparatu, szumy", na=False)]
        self.data = self.data[~self.data[self.type_col].str.contains("UCL", na=False)]


    def parse_date(self, x):
        try:
            return pd.to_datetime(x, format="%d.%m.%Y %H:%M")
        except ValueError:
            return pd.to_datetime(x, format="%Y-%m-%d %H:%M:%S")


    def patients_dfs(self):
        self.data[self.date_column] = self.data[self.date_column].apply(self.parse_date)
        self.data[self.implant_date_columnname] = pd.to_datetime(
            self.data[self.implant_date_columnname],
            format="%m/%d/%Y",
            errors='coerce'
        )

        self.data[self.implant_date_second_columnname] = pd.to_datetime(
            self.data[self.implant_date_second_columnname],
            format="%m/%d/%Y",
            errors='coerce'
        )
        
        self.data['date_year_month_day'] = self.data[self.date_column].dt.date
        self.group_columns = [self.patient_number_columnname] + ['date_year_month_day']

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
        elif typ in self.vibro_audiometry:
            return "vibro"
        

    def add_audiometry_group_and_ear_column(self):
        """Dodaje kolumnę GROUP do każdego mini_df."""
        for i, mini_df in enumerate(self.mini_dfs):
            mini_df["GROUP"] = mini_df[self.type_col].apply(self.assign_group)
            mini_df['EAR_SIDE'] = mini_df[self.earside_col].str.extract(r"(lewego|prawego)")
            mini_df['EAR_SIDE'] = mini_df['EAR_SIDE'].map({"lewego": "L", "prawego": "P"})


    def keep_first_delete_second(self, df, ear, columnname, merged_row):
        idx_first = df[df[columnname] == ear].index[0]
        #update values with merged from masking and not masking
        df.loc[idx_first] = merged_row
        #delete second row
        return df.loc[[idx_first]]


    def merge_masked_by_ear(self, group, ear):
        group = group[group['EAR_SIDE']== ear] #choose only one ear
        if group.shape[0] == 2:
            group_to_keep = group
        elif group.shape[0] > 2:
            if group[self.description_col].isna().any():
                for _, group_minutes in group.groupby(self.date_column):
                    if group_minutes[self.description_col].isna().any():
                        group_to_keep = group_minutes
            else:
                latest_date = group[self.date_column].max()
                group_to_keep = group[group[self.date_column] == latest_date]
        else:
            return group
        
        if group_to_keep.shape[0] > 1:
            merged_row = group_to_keep.iloc[0].combine_first(group_to_keep.iloc[1])
            return self.keep_first_delete_second(group_to_keep, ear, 'EAR_SIDE', merged_row)
        else:
            return group_to_keep


    def merge_masked(self):
        for i, mini_df in enumerate(self.mini_dfs):
            grouped = {g: d for g, d in mini_df.groupby("GROUP")} #group by air and bone audiometry, create a dict
            all_groups_df = pd.DataFrame()
            for key, group in grouped.items():
                ear_dfs = pd.DataFrame() #for each examination df for 2 ears
                for ear in self.ears:
                    ear_df = self.merge_masked_by_ear(group, ear) #merge ear in each audiometry type
                    ear_dfs = pd.concat([ear_dfs, ear_df], axis=0) #merge with other ear
                all_groups_df = pd.concat([all_groups_df, ear_dfs], axis=0) #merge bone and air audiometry
            self.mini_dfs[i] = all_groups_df
        print(f'Merging rows completed.')


    def create_air_audiometry(self, group, columns_to_fill):
        last_row = group.iloc[-1].copy()
        last_row[columns_to_fill] = 125
        last_row[self.type_col] = 'Air'
        last_row["GROUP"] = 'air'
        return last_row


    def find_last_existing_value(self, air_df, columns_to_fill):
        filled_last_idx = 0
        for i, column in enumerate(columns_to_fill):
            if air_df[column].notna().any():
                filled = True
            else:
                filled = False

            if filled:
                filled_last_idx = i
                filled_last_value = air_df.loc[air_df.index[0], column]

        return filled_last_idx, filled_last_value


    def fill_air_audiometry(self, air_df, columns_to_fill, filled_last_idx, vibro, filled_last_value=None, filling_limit=110):
        filled_previous = False
        filled = False
        fill = False

        if (not vibro) and (filled_last_value < filling_limit):
                return air_df
        else:
            for i, column in enumerate(columns_to_fill):
                if air_df[column].notna().any():
                    filled = True
                else:
                    filled = False

                if (filled is False) and (filled_previous is True) and (i>filled_last_idx):
                    air_df[column] = 125
                    fill = True
                    
                elif fill:
                    air_df[column] = 125

                filled_previous = filled

            return air_df


    def fill_ending_values(self, columns_to_fill, filling_limit):
        valid_mini_dfs = [] 
        for i, mini_df in enumerate(self.mini_dfs):
            ears_grouped = {g: d for g, d in mini_df.groupby("EAR_SIDE")}
            for key, group in ears_grouped.items():
                audiometry_type_grouped = {g: d for g, d in group.groupby("GROUP")}
                bone_df = audiometry_type_grouped.get('bone')
                air_df = audiometry_type_grouped.get('air')
                vibro_df = audiometry_type_grouped.get('vibro')

                if (air_df is None) and (vibro_df is not None):
                    last_row = self.create_air_audiometry(group, columns_to_fill)
                    group = pd.concat([group, pd.DataFrame([last_row])], ignore_index=True)
                    
                elif (bone_df is None) and (vibro_df is not None):
                    filled_last_idx, _ = self.find_last_existing_value(air_df, columns_to_fill)
                    air_df = self.fill_air_audiometry(air_df, columns_to_fill=columns_to_fill, filled_last_idx=filled_last_idx, vibro=True)
                    group.loc[air_df.index, :] = air_df
                elif (air_df is None):
                    group=None
                    continue
                else:
                    filled_last_idx, filled_last_value = self.find_last_existing_value(air_df, columns_to_fill)
                    air_df = self.fill_air_audiometry(air_df, columns_to_fill=columns_to_fill, filled_last_idx=filled_last_idx, vibro=False, filled_last_value=filled_last_value, filling_limit=filling_limit)
                    group.loc[air_df.index, :] = air_df

                ears_grouped[key] = group 

            mini_df = pd.concat(ears_grouped.values(), ignore_index=True)       
            valid_mini_dfs.append(mini_df)
        self.mini_dfs = valid_mini_dfs


    def mark_implanted_ear(self):
        for i, mini_df in enumerate(self.mini_dfs):
            ears_grouped = {g: d for g, d in mini_df.groupby("EAR_SIDE")}
            implanted_ears = set()
            
            for ear in self.ears:
                if ear not in ears_grouped:
                    continue

                group = ears_grouped[ear]
                indices = group.index
                indices_other = mini_df[mini_df["EAR_SIDE"] != ear].index

                implanted = False
                if (group[self.implant_ear_columnname] == ear).any():
                    mask = group[self.date_column] > group[self.implant_date_columnname]
                    if mask.all():
                        implanted = True
                elif (group[self.implant_ear_second_columnname] == ear).any():
                    mask = group[self.date_column] > group[self.implant_date_second_columnname]
                    if mask.all():
                        implanted = True
                else:
                    mask = False

                if implanted and len(implanted_ears) == 0:
                    self.mini_dfs[i].loc[indices_other, 'PO_IMPLANTACJI'] = 'drugie_implantowane'
                    self.mini_dfs[i].loc[indices, 'PO_IMPLANTACJI'] = mask
                elif len(implanted_ears) > 0 and not implanted:
                    pass
                else:
                    self.mini_dfs[i].loc[indices, 'PO_IMPLANTACJI'] = mask
                    
                if implanted:
                    implanted_ears.add(ear)


    def delete_implanted_ear(self):
        cleaned = []
        for mini_df in self.mini_dfs:
            df = mini_df[~(mini_df['PO_IMPLANTACJI'] == True)]
            if not df.empty:
                cleaned.append(df)
        self.mini_dfs = cleaned

    
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


    def define_symmetry(self, first_symmetry_columns, second_symmetry_columns, threshold_def1, threshold_def2, suffix="_diff"):
        for i, mini_df in enumerate(self.mini_dfs):
            grouped = {g: d for g, d in mini_df.groupby("GROUP")}
            for key in grouped:
                if key == "air":  #only for air audiometry
                    group = grouped[key] #group inherits indexes after mini_df
                    if group.shape[0] != 2:
                        group.loc[:, 'SYMETRIA'] = "brak_obl" #is there is only one ear, the calculation is not possible
                    else:
                        diff_def1 = self.compute_diff(group, first_symmetry_columns, suffix)
                        diff_def2 = self.compute_diff(group, second_symmetry_columns, suffix) #differences for 2 definitions

                        group['SYMETRIA_1_DEF'] = self.check_symmetry_def1(diff_def1, threshold_def1)
                        group['SYMETRIA_2_DEF'] = self.check_symmetry_def2(diff_def2, threshold_def2) 
                        group['SYMETRIA'] = group.apply(self.combine_sym, axis=1) #checking conditions

                        for col in diff_def1.columns:
                            group[col] = diff_def1[col].iloc[0]
                            self.mini_dfs[i][col] = group[col] #add column with differences

                        self.mini_dfs[i]['SYMETRIA_1_DEF'] = group['SYMETRIA_1_DEF']
                        self.mini_dfs[i]['SYMETRIA_2_DEF'] = group['SYMETRIA_2_DEF']
                    self.mini_dfs[i]['SYMETRIA'] = group['SYMETRIA']



    def calculate_mean_ear_pta(self, PTA2_columns, PTA4_columns, lfPTA_columns_1, lfPTA_columns_2, hf_columns):
        numeric_cols = PTA2_columns + PTA4_columns + hf_columns
        text_cols = [col for col in self.data.columns if col not in numeric_cols]
        #valid_mini_dfs = [] 

        for i, mini_df in enumerate(self.mini_dfs):
            grouped = {g: d for g, d in mini_df.groupby("GROUP")} #group by air and bone

            if "air" not in grouped:
                print(f"[Info] Skipping mini_df {i}: no 'air' group found.")
                continue

            for key, group in grouped.items():
                if key == "air": #calculate only for air audiometry
                    sym = grouped["air"].iloc[0]['SYMETRIA'] #symmetry condition
                    if sym == 1 and group.shape[0] == 2:
                        df_source = pd.DataFrame({
                            **{col: [group[col].mean()] for col in numeric_cols},
                            **{col: [group[col].iloc[0]] for col in text_cols}
                        })  #calculate mean from row in group in case of symmetry
                    else:
                        df_source = group #else, it stays as 2 distinct rows

                    pta2_mean, pta4_mean, ptalf_1_mean, ptalf_2_mean, ptahf_mean = self.calculate_pta(df_source, PTA2_columns, PTA4_columns, lfPTA_columns_1, lfPTA_columns_2, hf_columns) 

                    if sym == 1 and group.shape[0] == 2:
                        pta2_mean = pta2_mean.item() #because it returns series from two rows
                        pta4_mean = pta4_mean.item()
                        ptalf_1_mean = ptalf_1_mean.item()
                        ptalf_2_mean = ptalf_2_mean.item()
                        ptahf_mean = ptahf_mean.item()

                    group.loc[:, 'PTA2'] = pta2_mean #assign the values
                    group.loc[:, 'PTA4'] = pta4_mean
                    group.loc[:, 'lfPTA_1'] = ptalf_1_mean
                    group.loc[:, 'lfPTA_2'] = ptalf_2_mean
                    group.loc[:, 'hfPTA'] = ptahf_mean

                    self.mini_dfs[i]['PTA2'] = group['PTA2']
                    self.mini_dfs[i]['PTA4'] = group['PTA4']
                    self.mini_dfs[i]['lfPTA_1'] = group['lfPTA_1']
                    self.mini_dfs[i]['lfPTA_2'] = group['lfPTA_2']
                    self.mini_dfs[i]['hfPTA'] = group['hfPTA']
                    #valid_mini_dfs.append(mini_df)
        #self.mini_dfs = valid_mini_dfs
        print('PTA calculation completed.')


    def calculate_pta(self, df, PTA2_columns, PTA4_columns, lfPTA_columns_1, lfPTA_columns_2, hf_columns):
        pta2_mean = df[PTA2_columns].mean(axis=1, skipna=False).round(0)
        pta4_mean = df[PTA4_columns].mean(axis=1, skipna=False).round(0)
        ptalf_1_mean = df[lfPTA_columns_1].mean(axis=1, skipna=False).round(0)
        ptalf_2_mean = df[lfPTA_columns_2].mean(axis=1, skipna=False).round(0)
        ptahf_mean = df[hf_columns].mean(axis=1, skipna=False).round(0)
        return pta2_mean, pta4_mean, ptalf_1_mean, ptalf_2_mean, ptahf_mean
    

    def map_hearing_level(self, hearing_levels, value):
        for level in hearing_levels:
            if value <= level["max"]:
                return level["label"]


    def classificate_hearing_loss(self, biap_hearing_levels, asha_hearing_levels):
        for i, mini_df in enumerate(self.mini_dfs):
            mini_df["BIAP_PTA2"] = mini_df['PTA2'].apply(lambda x: self.map_hearing_level(biap_hearing_levels, x))
            mini_df["BIAP"] = mini_df['PTA4'].apply(lambda x: self.map_hearing_level(biap_hearing_levels, x))
            mini_df["ASHA"] = mini_df['PTA4'].apply(lambda x: self.map_hearing_level(asha_hearing_levels, x))
            mini_df["BIAP_lfPTA_1"] = mini_df['lfPTA_1'].apply(lambda x: self.map_hearing_level(biap_hearing_levels, x))
            mini_df["BIAP_lfPTA_2"] = mini_df['lfPTA_2'].apply(lambda x: self.map_hearing_level(biap_hearing_levels, x))
            mini_df["BIAP_hfPTA"] = mini_df['hfPTA'].apply(lambda x: self.map_hearing_level(biap_hearing_levels, x))

        print("Hearing loss calculation completed")
        

    def check_condition_df(self, df, cond):
        ops = {
            ">=": operator.ge,
            ">": operator.gt,
            "<=": operator.le,
            "<": operator.lt,
        }
        op = cond["operator"]
        threshold = cond["value"]
        
        if "column" in cond:
            series = df[cond["column"]]
            return ops[op](series, threshold)
        
        col1, col2 = cond["columns"]
        diff = df[col1] - df[col2]

        if cond["operator"] == "abs<=":
            return diff.abs() <= cond["value"]
        else:
            return ops[op](diff, threshold)


    def match_audiogram_type(self, audiogram_types_criteria):
        for i, mini_df in enumerate(self.mini_dfs):
            ears_grouped = {g: d for g, d in mini_df.groupby("EAR_SIDE")}
            for key, group in ears_grouped.items():

                for type_name, rule in audiogram_types_criteria.items():
                    ok = True
                    for cond in rule["conditions"]:
                        mask = self.check_condition_df(group, cond).all()
                        if not mask.any():
                            ok = False
                            break
                    if ok:
                        self.mini_dfs[i].loc[group.index, 'TYPE_ZONE_1'] =  type_name
                        break

        print("Audiogram types matched")

    def check_threshold(self, threshold, value):
        if value < threshold:
            return 1
        elif value >= threshold:
            return 0
        

    def hearing_loss_type_cond1(self, df, threshold):
        df["PTA4_condition"] = df['PTA4'].apply(lambda x: self.check_threshold(threshold, x))
        df["hfPTA_condition"] = df['hfPTA'].apply(lambda x: self.check_threshold(threshold, x))
        return df
    

    def hearing_loss_type_cond2(self, group, bone_mean_columns, threshold):
        group = group.copy()
        group['bone_mean'] = pd.Series(dtype='object')
        group['bone_mean_condition'] = pd.Series(dtype='object')

        for idx, row in group.iterrows():
            if row[bone_mean_columns].isna().all():
                group.loc[idx, 'bone_mean'] = "brak_obl"
                group.loc[idx, 'bone_mean_condition'] = "brak_obl"
            else:
                mean_val = round(row[bone_mean_columns].mean(), 1)
                group.loc[idx, 'bone_mean'] = mean_val
                group.loc[idx, 'bone_mean_condition'] = self.check_threshold(threshold, mean_val)

        return group


    def hearing_type_pta_and_bone_audiometry(self, threshold, bone_mean_columns, bone_hf_mean_columns):
        for i, mini_df in enumerate(self.mini_dfs):
            mini_df = self.hearing_loss_type_cond1(mini_df, threshold)
            grouped = {g: d for g, d in mini_df.groupby("GROUP")}
            for key, group in grouped.items():
                if key == "bone":
                    group = self.hearing_loss_type_cond2(group, bone_mean_columns, threshold)
                    self.mini_dfs[i]['bone_mean'] = group['bone_mean']
                    self.mini_dfs[i]['bone_mean_condition'] = group['bone_mean_condition']
                    group = self.hearing_loss_type_cond2(group, bone_hf_mean_columns, threshold)
                    self.mini_dfs[i]['bone_mean_hf'] = group['bone_mean']
                    self.mini_dfs[i]['bone_mean_condition_hf'] = group['bone_mean_condition']
                    continue
                elif key == "vibro":
                    self.mini_dfs[i].loc[group.index, 'bone_mean_condition'] = 0
                    self.mini_dfs[i].loc[group.index, 'bone_mean_condition_hf'] = 0


    def check_differences_opt1_zero(self, diff_df, value = 0, expected_length=4):
        diff_df = diff_df.dropna(axis=1, how='all')
        if (diff_df.shape[1]!=expected_length):
            return 'brak_obl'
        elif (diff_df.iloc[0]!=0).sum() == value:
            return 1
        else:
            return 0
        

    def check_differences_opt1(self, diff_df, threshold=10, how_many=3, expected_length=4):
        diff_df = diff_df.dropna(axis=1, how='all')
        if (diff_df.shape[1]<expected_length):
            return "brak_obl"

        row = diff_df.iloc[0]
        exceeds = (row.abs() >= threshold).astype(int)

        max_run = 0
        current_run = 0
        for val in exceeds:
            if val:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        return int(max_run >= how_many)
    
    

    def hearing_type_differences_between_audiometries(self, first_opt_columns, threshold, how_many_values):
        for i, mini_df in enumerate(self.mini_dfs):
            ears_grouped = {g: d for g, d in mini_df.groupby("EAR_SIDE")}
            for key, group in ears_grouped.items():
                audiometry_type_grouped = {g: d for g, d in group.groupby("GROUP")}
                bone_df = audiometry_type_grouped.get('bone')
                air_df = audiometry_type_grouped.get('air')
                vibro_df = audiometry_type_grouped.get('vibro')

                frames = [df for df in [bone_df, air_df] if df is not None]

                if frames:
                    bone_air_df = pd.concat(frames, ignore_index=True)

                diff_opt_1 = self.compute_diff(bone_air_df, first_opt_columns, suffix="_diff_first_opt")

                group[f'first_option_{threshold}_diff'] = self.check_differences_opt1(diff_opt_1, threshold=threshold, how_many=how_many_values, expected_length=len(first_opt_columns))
                group[f'15_diff'] = self.check_differences_opt1(diff_opt_1, threshold=15, how_many=1, expected_length=1)

                self.mini_dfs[i].loc[group.index, f'first_option_{threshold}_diff'] = group[f'first_option_{threshold}_diff'].astype('object')
                self.mini_dfs[i].loc[group.index, f'15_diff'] = group[f'15_diff'].astype('object')

                col1 = pd.to_numeric(group[f'first_option_{threshold}_diff'], errors='coerce')
                col2 = pd.to_numeric(group['15_diff'], errors='coerce')

                rezerwa = np.where(
                    col1.eq(1) | col2.eq(1), 1,    #jeśli którakolwiek kolumna ma 1 → 1
                    np.where(
                        col1.isna() & col2.isna(), np.nan,  #jeśli obie NaN → NaN (później zamienimy na brak_obl)
                        0                              #w pozostałych przypadkach → 0
                    )
                )
                if np.isnan(rezerwa).any() and vibro_df is not None:
                    rezerwa = np.nan_to_num(rezerwa, nan=0) 

                self.mini_dfs[i].loc[group.index, 'REZERWA'] = pd.Series(rezerwa, index=group.index).replace({np.nan: "brak_obl"})

                if not diff_opt_1.empty:
                    for col in diff_opt_1.columns:
                        group[col] = diff_opt_1[col].iloc[0]
                        self.mini_dfs[i].loc[group.index, col] = group[col] #add column with differences


    def classificate_hearing_loss_type(self, criteria):
        for i, mini_df in enumerate(self.mini_dfs):
            grouped = {g: d for g, d in mini_df.groupby("GROUP")}
            #jeśli nie ma dwóch grup (air + bone), nie określamy typu
            if len(grouped) < 2:
                self.mini_dfs[i].loc[:, 'hearing_type'] = "nie okreslono"
                continue

            for ear in self.ears:
                ear_assigned = False
                for loss_type, rules in criteria.items():
                    match = True
                    for key, conditions in rules.items():

                        try:
                            group = grouped[key]
                        except KeyError:
                            #print(f"[INFO] Brak grupy '{key}' w danych, pomijam.")
                            match = False
                            break

                        ear_row = group[group['EAR_SIDE'] == ear]

                        if ear_row.empty:
                            match = False
                            break

                        for col, expected in conditions.items():
                            if ear_row[col].item() != expected:
                                match = False

                    if match:
                        #przypisanie typu ubytku do wszystkich wierszy tego ucha
                        indices = grouped['air'][grouped['air']['EAR_SIDE'] == ear].index
                        self.mini_dfs[i].loc[indices, 'hearing_type'] = loss_type
                        ear_assigned = True
                        break  #jeśli dopasowano typ, nie sprawdzaj innych

                if not ear_assigned:
                    #jeśli żaden typ nie pasuje, oznaczamy jako "nie określono"
                    indices = mini_df[mini_df['EAR_SIDE'] == ear].index
                    self.mini_dfs[i].loc[indices, 'hearing_type'] = "zaden typ nie pasuje"


    def save_processed_df(self, output_path):
        merged_df = pd.concat(self.mini_dfs, ignore_index=True)
        merged_df[self.date_column] = merged_df[self.date_column].dt.strftime("%d.%m.%Y %H:%M")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        merged_df.to_csv(f'{output_path}audiometry_{self.tonal_suffix}.csv', index=False, sep=";", encoding="utf-8-sig")
        print(f'Saving to {output_path}audiometry_{self.tonal_suffix}.csv completed.')


