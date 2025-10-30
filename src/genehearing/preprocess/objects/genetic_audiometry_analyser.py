import pandas as pd
from genehearing.preprocess.objects.tonal_audiometry import TonalAudiometry

class GenehearingAnalyser(TonalAudiometry):
        def __init__(self, path, 
                  tonal_suffix,
                  columnnames,
                  air_audiometry=["AirMask", "Air"],
                  bone_audiometry=["BoneMask", "Bone"]):
            super().__init__(path, tonal_suffix, columnnames, air_audiometry, bone_audiometry)