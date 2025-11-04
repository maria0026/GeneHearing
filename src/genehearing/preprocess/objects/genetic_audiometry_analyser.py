import pandas as pd
from genehearing.preprocess.objects.tonal_audiometry import TonalAudiometry

class GenehearingAnalyser(TonalAudiometry):
        def __init__(self, path, 
                  tonal_suffix,
                  implants_datapath,
                  columnnames,
                  implant_columnnames,
                  air_audiometry=["AirMask", "Air"],
                  bone_audiometry=["BoneMask", "Bone"]):
            super().__init__(path, tonal_suffix, implants_datapath, columnnames, implant_columnnames, air_audiometry, bone_audiometry)

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