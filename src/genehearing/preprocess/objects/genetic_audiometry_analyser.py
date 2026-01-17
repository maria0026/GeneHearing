import pandas as pd
import os
from genehearing.preprocess.objects.tonal_audiometry import TonalAudiometry

class GenehearingAnalyser(TonalAudiometry):
        def __init__(self, path, 
                  tonal_suffix,
                  implants_datapath,
                  columnnames,
                  implant_columnnames,
                  air_audiometry=["AirMask", "Air"],
                  bone_audiometry=["BoneMask", "Bone"],
                  vibro_audiometry=["Vibro", "VibroMask"]):
            super().__init__(path, tonal_suffix, implants_datapath, columnnames, implant_columnnames, air_audiometry, bone_audiometry, vibro_audiometry)

        def choose_first_examination(self):
            id = 0
            for i, mini_df in enumerate(self.mini_dfs):
                id_new = mini_df[self.patient_number_columnname].values[0]
                #print(id, id_new)
                if id_new == id:
                    self.mini_dfs[i]['IF_FIRST'] = 0
                else:
                    self.mini_dfs[i]['IF_FIRST'] = 1
                id = id_new


        def create_dataframe_for_merging(self, biap_columns, output_path):
            biap_names = []
            for ear in self.ears:
                for biap_column in biap_columns:
                    biap_names.append(f"{ear}_{biap_column}")
            for i, mini_df in enumerate(self.mini_dfs):
                ears_grouped = {g: d for g, d in mini_df.groupby("EAR_SIDE")}
                for ear in self.ears:
                    if ear not in ears_grouped:
                        continue

                    group = ears_grouped[ear]
                    for biap_column in biap_columns:
                        biap_values = group[biap_column].dropna()
                        if biap_values.empty:
                            self.mini_dfs[i][f'{ear}_{biap_column}'] = None
                        else:
                            self.mini_dfs[i][f'{ear}_{biap_column}'] = biap_values.iloc[0]

                    hearing_type_vals = group['HEARING_TYPE'].dropna()
                    if hearing_type_vals.empty:
                        self.mini_dfs[i][f'{ear}_HEARING_TYPE'] = None
                    else:
                        self.mini_dfs[i][f'{ear}_HEARING_TYPE'] = hearing_type_vals.iloc[0]

                    
                indeks = mini_df.index[0]
                self.mini_dfs[i] = mini_df.iloc[[indeks]]

            merged_df = pd.concat(self.mini_dfs, ignore_index=True)
            merged_df[self.date_column] = merged_df[self.date_column].dt.strftime("%d.%m.%Y %H:%M")
            self.new_df = merged_df.loc[:, [self.patient_number_columnname, self.date_column, 'IF_FIRST', *biap_names, 'L_HEARING_TYPE', 'P_HEARING_TYPE']]
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            self.new_df.to_csv(f'{output_path}audiometry_{self.tonal_suffix}_summarized.csv', index=False)
            print(f'Saving to {output_path}audiometry_{self.tonal_suffix}_summarized.csv completed.')



        def create_disinct_datasets(self, hearing_loss_dict, biap_columns, output_path):
            for loss_type, rules in hearing_loss_dict.items():
                df_filtered = self.new_df[(self.new_df[biap_columns[0]]==rules[0]) & (self.new_df[biap_columns[1]]==rules[1])]
                df_filtered.to_csv(f'{output_path}audiometry_{loss_type}.csv', index=False)
                print(f'Saving to {output_path}audiometry_{loss_type}.csv completed.')