import genehearing.common.tools as tools
from genehearing.preprocess.objects.abr import AuditoryBrainstemResponse

def main():

    config = tools.load_config("config.yaml")
    tonaldataname=config["tonaldataname"]
    tonal_suffix = tonaldataname.split("_")[-1]
    audiometry_datapath = config["datacalculationsdirectory"] + tonaldataname + '_' + config['genetic_name']+'.csv'
    
    abr_datapath = config["datarawdirectory"] + config['abr'] + '.csv'
    genetic_datapath = config["datarawdirectory"] + config['genetic_dataname_all'] + '.csv'
    output_path = config["dataprocesseddirectory"] + config['genetic_name'] + '_abr'+'.csv'


    abr_analyser = AuditoryBrainstemResponse(abr_datapath, 
                                    genetic_datapath,
                                    columnnames={"columns_to_fill": config["abr_columns_to_fill"],
                                                 "optional_columns": config["abr_optional_columns"],
                                                 "additional_column": config["abr_additional_column"]},
                                    output_path=output_path,
                                    match_column=config['pesel_columnname'])
    
    abr_analyser.merge_genetic_abr()
    abr_analyser.map_no_response()
    abr_analyser.replace_values()

    PTA_columns = {'PTA2': config["pta_abr_columns"]["PTA2"],
                    'PTA4': config["pta_abr_columns"]["PTA4"],
                  }
    abr_analyser.calculate_PTA(PTA_columns)

    
    #merge genetic and abr, then concat audiometry tonal

if __name__=="__main__":
    main()