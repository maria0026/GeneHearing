from genehearing.preprocess.objects.tonal_audiometry import TonalAudiometry
import genehearing.common.tools as tools

def main():

    config = tools.load_config("config.yaml")
    tonaldataname=config["tonaldataname"]
    tonal_suffix = tonaldataname.split("_")[-1]
    tonal_audiometry_datapath = config["datarawdirectory"] + tonaldataname + '_' + config['genetic_name']+'.csv'
    implants_datapath = config["datarawdirectory"] + config['implants'] + '.csv'


    tonal_audiometry_processor = TonalAudiometry(tonal_audiometry_datapath, 
                                                 tonal_suffix, 
                                                 implants_datapath,
                                                 columnnames={'patient_number_columnname': config["patient_number_columnname"],
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
                                                vibro_audiometry=config['vibro_audiometry']
                                                 )
    tonal_audiometry_processor.merge_implants()
    tonal_audiometry_processor.filter_audiometry_type()
    tonal_audiometry_processor.patients_dfs()
    tonal_audiometry_processor.add_audiometry_group_and_ear_column()
    tonal_audiometry_processor.merge_masked()

    tonal_audiometry_processor.fill_ending_values(config["columns_to_fill_standard"], config["columns_to_fill_all"], config['filling_limit'], config['halfoctave_columns'])
    tonal_audiometry_processor.mark_implanted_ear()
    #tonal_audiometry_processor.delete_implanted_ear()


    first_symmetry_columns = config["first_symmetry_columns"]
    second_symmetry_columns = config["second_symmetry_columns"]

    tonal_audiometry_processor.define_symmetry(first_symmetry_columns, second_symmetry_columns, config["threshold_def1"], config["threshold_def2"])
    PTA_columns = { 'lfPTA_1': config["pta_columns"]["lfPTA_1"],
                    'lfPTA_2': config["pta_columns"]["lfPTA_2"],
                    'PTA2': config["pta_columns"]["PTA2"],
                    'PTA4': config["pta_columns"]["PTA4"],
                    'hfPTA': config["pta_columns"]["hfPTA"],
                    'lf_zone_PTA': config["pta_columns"]["lfzone"],
                    'mf_zone_PTA': config["pta_columns"]["mfzone"],
                    'hf_zone_PTA': config["pta_columns"]["hfzone"]
                    }
    
    tonal_audiometry_processor.calculate_mean_ear_pta(PTA_columns)

    tonal_audiometry_processor.classificate_hearing_loss(PTA_columns, config["biap_hearing_levels"], config["asha_hearing_levels"])
    tonal_audiometry_processor.match_audiogram_type(config['audiogram_types_criteria_zone_1'], config['audiogram_types_criteria_zone_2'],
                                                    config['audiogram_types_criteria_zones'])

    tonal_audiometry_processor.hearing_type_pta_and_bone_audiometry(config["pta_threshold"], config["bone_all_mean_columns"], config["bone_hf_all_mean_columns"])
    
    tonal_audiometry_processor.hearing_type_differences_between_audiometries(config['reserve_columns'], threshold=config['reserve_threshold'], how_many_values=config['reserve_how_many'])
    tonal_audiometry_processor.classificate_hearing_loss_type(config["hearing_loss_criteria"])


    tonal_audiometry_processor.save_processed_df(config["dataprocesseddirectory"])


if __name__=="__main__":
    main()