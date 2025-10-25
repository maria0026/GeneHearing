from genehearing.preprocess.objects.tonal_audiometry import TonalAudiometry
import genehearing.common.tools as tools

def main():

    config = tools.load_config("config.yaml")
    tonaldataname=config["tonaldataname"]
    tonal_suffix = tonaldataname.split("_")[-1]
    tonal_audiometry_datapath = config["datarawdirectory"] + tonaldataname + '.csv'

    tonal_audiometry_processor = TonalAudiometry(tonal_audiometry_datapath, 
                                                 tonal_suffix, 
                                                 columnnames={'pesel_columnname': config["pesel_columnname"],
                                                                'audiometry_earside_columnname': config['audiometry_earside_columnname'],
                                                                'date_column': config['date_column'],
                                                                'type_column': config['audiometry_type_columnname']
                                                               },
                                                 air_audiometry=config['air_audiometry'],
                                                 bone_audiometry=config['bone_audiometry']
                                                 )
    tonal_audiometry_processor.filter_audiometry_type()
    tonal_audiometry_processor.patients_dfs()
    tonal_audiometry_processor.add_audiometry_group_column()
    tonal_audiometry_processor.merge_masked()

    PTA2_columns = config["pta_columns"]["PTA2"]
    PTA4_columns = config["pta_columns"]["PTA4"]
    hfPTA_columns = config["pta_columns"]["hfPTA"]

    first_symmetry_columns = config["first_symmetry_columns"]
    second_symmetry_columns = config["second_symmetry_columns"]

    tonal_audiometry_processor.define_symmetry(first_symmetry_columns, second_symmetry_columns, config["threshold_def1"], config["threshold_def2"])
    tonal_audiometry_processor.calculate_mean_ear_pta(PTA2_columns, PTA4_columns, hfPTA_columns)

    tonal_audiometry_processor.classificate_hearing_loss(config["biap_hearing_levels"], config["asha_hearing_levels"])

    tonal_audiometry_processor.hearing_type_pta_and_bone_audiometry(config["pta_threshold"], config["bone_all_mean_columns"])
    
    tonal_audiometry_processor.hearing_type_differences_between_audiometries(config['first_opt_columns'], threshold=config['first_opt_threshold'], how_many_values=config['first_opt_how_many'])
    tonal_audiometry_processor.classificate_hearing_loss_type(config["hearing_loss_criteria"])

    #tonal_audiometry_processor.classificate_hearing_loss_type_normal()
    #tonal_audiometry_processor.classificate_hearing_loss_type_conductive()
    #tonal_audiometry_processor.classificate_hearing_loss_type_receiving()
    #tonal_audiometry_processor.classificate_hearing_loss_type_mixed()

    tonal_audiometry_processor.save_processed_df(config["dataprocesseddirectory"])

    #tonal_audiometry_processor.select_better_air_pta()
    #tonal_audiometry_processor.save_processed_to_mri_df(config["dataprocesseddirectory"])


if __name__=="__main__":
    main()