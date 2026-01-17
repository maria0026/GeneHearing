import genehearing.common.tools as tools
from genehearing.preprocess.objects.genetic_audiometry_analyser import GenehearingAnalyser

def main():
    config = tools.load_config()
    tonaldataname=config["tonaldataname"]
    tonal_suffix = tonaldataname.split("_")[-1]
    datapath = config["dataprocesseddirectory"] + tonaldataname+'.csv'
    implants_datapath = config["datarawdirectory"] + config['implants'] + '.csv'


    genehearing_analyser = GenehearingAnalyser(datapath, 
                                                tonal_suffix, 
                                                implants_datapath,
                                                columnnames={'patient_number_columnname': config["patient_number_columnname"],
                                                            'patient_id_columnname': config["patient_id_columnname"],
                                                            'audiometry_earside_columnname': config['audiometry_earside_columnname'],
                                                            'date_column': config['date_column'],
                                                            'type_column': config['audiometry_type_columnname'],
                                                            'description_column': config['description_columnname'],
                                                            'genetic_patient_id_column': config['patient_identifier_columnname']
                                                            },
                                                implant_columnnames={"patient_identifier_columnname": config['patient_identifier_columnname'],
                                                                'age_of_occurence_columnname': config['age_of_occurence_columnname'],
                                                                'age_of_recognition_columnname': config['age_of_recognition_columnname'],
                                                                'implant_date_columnname': config['implant_date_columnname'],
                                                                'implant_ear_columnname': config['implant_ear_columnname'],
                                                                'second_implant_date_columnname': config['second_implant_date_columnname'],
                                                                'second_implant_ear_columnname':  config['second_implant_ear_columnname']
                                                            },
                                                air_audiometry=config['air_audiometry'],
                                                bone_audiometry=config['bone_audiometry'],
                                                vibro_audiometry=config['vibro_audiometry'])
    

    genehearing_analyser.patients_dfs()
    genehearing_analyser.choose_first_examination()

    biap_columns = ['BIAP_PTA4', 'BIAP_LF_ZONE_PTA', 'BIAP_MF_ZONE_PTA', 'BIAP_HF_ZONE_PTA']
    genehearing_analyser.create_dataframe_for_merging(biap_columns, config["datacalculationsdirectory"])
    biap_columns_sel = ['L_BIAP_HF_ZONE_PTA', 'P_BIAP_HF_ZONE_PTA']
    genehearing_analyser.create_disinct_datasets(config['Datasets'], biap_columns_sel, config["datacalculationsdirectory"])
    genehearing_analyser.save_processed_df(config["datacalculationsdirectory"])


if __name__=="__main__":

    main()