from neurohearing.preprocess.objects.mri_morphometrics import MRI_morphometics
from brain_age.utils import explain
import neurohearing.common.tools as tools
import pandas as pd
import argparse
from tqdm import tqdm

def main(args):
    config = tools.load_config()
    mri_suffix = "mri"
    output_path_mri = config["dataprocesseddirectory"] +  config["mri_dataname"] + '.csv'
    #output_path_merged = config["dataprocesseddirectory"] + args.mri_audiometry_merged_filename +'.csv'
    df_scores = pd.DataFrame()

    mri_morpohometrics_train = MRI_morphometics(output_path_mri,
                                          mri_suffix,
                                          config)

    mri_df = mri_morpohometrics_train.data_mri
    numeric_cols = mri_df.select_dtypes(include='number').columns
    all_results = []
    for column in tqdm(numeric_cols, desc="Calculating trends"):

        model, coefficients = explain.calculate_trends(mri_df, column, args.label_name)
        white_test_pvalue = explain.white_test(mri_df, column, args.label_name, model)
        statistics_scores = explain.scores(mri_df, column, args.label_name, model)

        row_dict = {
                "column": column,
                "1": coefficients[0] if len(coefficients) > 0 else None,
                "x": coefficients[1] if len(coefficients) > 1 else None,
                "x^2": coefficients[2] if len(coefficients) > 2 else None,
                "White test p-value": white_test_pvalue,
                **statistics_scores  # mean, std, skewness, kurtosis, itd.
            }
        
        quantiles_array, indices_array = explain.calculate_quantiles(mri_df, column, args.label_name, args.quantiles, model)

        for idx, val in zip(indices_array, quantiles_array):
            row_dict[idx] = val
        all_results.append(row_dict)

    df_scores = pd.DataFrame(all_results)
    print(df_scores)

    scores_results_directory = config["resultsdirectory"] + "mri_trends.csv"
    df_scores.set_index(numeric_cols, inplace=True)
    df_scores.to_csv(scores_results_directory)

    #plt.scatter(mri_morpohometrics.data_mri['PTA2'], mri_morpohometrics.data_mri['A2009-ctx-lh-G_and_S_frontomargin_GrayVol'])
    #plt.show()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Parser for audiometry data analyzer")
    parser.add_argument("--mri_audiometry_merged_filename", 
                        nargs="?", 
                        default="audiometry_tonal_mri_merged",
                        help="Filename for audiometry mri merged")
    parser.add_argument("--label_name",
                        nargs="?",
                        default="age",
                        help="Predicted parameters, list", 
                        type=str)
    parser.add_argument("--quantiles",
                        nargs="?",
                        default=[0.005, 0.01, 0.02, 0.05],
                        help="Quantiles for scores",
                        type=list)
    args = parser.parse_args()
    main(args)