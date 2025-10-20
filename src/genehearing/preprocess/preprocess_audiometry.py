from genehearing.preprocess.objects.tonal_audiometry import TonalAudiometry
import genehearing.common.tools as tools

def main():

    config = tools.load_config()
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
    tonal_audiometry_processor.merge_mask()

    PTA2_columns = config["pta_columns"]["PTA2"]
    PTA4_columns = config["pta_columns"]["PTA4"]
    hfPTA_columns = config["pta_columns"]["hfPTA"]

    first_symmetry_columns = config["first_symmetry_columns"]
    second_symmetry_columns = config["second_symmetry_columns"]

    tonal_audiometry_processor.define_symmetry(first_symmetry_columns, second_symmetry_columns)
    
    tonal_audiometry_processor.calculate_mean_ear_pta(PTA2_columns, PTA4_columns, hfPTA_columns)
    tonal_audiometry_processor.save_processed_df(config["dataprocesseddirectory"])

    #tonal_audiometry_processor.select_better_air_pta()
    #tonal_audiometry_processor.save_processed_to_mri_df(config["dataprocesseddirectory"])


if __name__=="__main__":
    main()