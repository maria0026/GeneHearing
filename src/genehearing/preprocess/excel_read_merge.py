from genehearing.preprocess.objects.preprocessing_core import CSVProcessor
import genehearing.common.tools as tools


def main():

    config = tools.load_config()
    tonaldataname=config["tonaldataname"]
    tonal_suffix = tonaldataname.split("_")[-1]
    tonal_audiometry_datapath = config["datarawdirectory"] + tonaldataname + '.csv'
    genetic_datapath = config["datarawdirectory"] + config['genetic_dataname_all'] + '.csv'
    

    tonal_audiometry_processor = CSVProcessor(tonal_audiometry_datapath, tonal_suffix, genetic_datapath, output_path=config["datarawdirectory"], match_column=config['pesel_columnname']) 
    tonal_audiometry_processor.read_merge_genetic_audiometry()
    tonal_audiometry_processor.save_merged()



if __name__=="__main__":
    main()