import genehearing.common.tools as tools
from genehearing.preprocess.objects.tonal_audiometry import TonalAudiometry
from genehearing.preprocess.objects.genetic_audiometry_analyser import GenehearingAnalyser
import pandas as pd
import argparse
from tqdm import tqdm

def main():
    config = tools.load_config()
    tonaldataname=config["tonaldataname"]
    tonal_suffix = tonaldataname.split("_")[-1]
    datapath = config["dataprocesseddirectory"] + tonaldataname + '_' + config['genetic_name']+'.csv'


    genehearing_analyser = GenehearingAnalyser(datapath, 
                                                tonal_suffix, 
                                                columnnames={'patient_number_columnname': config["patient_number_columnname"],
                                                            'audiometry_earside_columnname': config['audiometry_earside_columnname'],
                                                            'date_column': config['date_column'],
                                                            'type_column': config['audiometry_type_columnname'],
                                                            'description_column': config['description_columnname']
                                                            },
                                                air_audiometry=config['air_audiometry'],
                                                bone_audiometry=config['bone_audiometry'])
if __name__=="__main__":

    main()