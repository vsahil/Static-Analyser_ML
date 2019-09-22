import pandas as pd
from tabulate import tabulate

latex_file = "presentation_results_fse"
def create_latex_from_table(filename:str, table:pd.DataFrame, create_doc:bool):
    if create_doc:
        latex = """\\documentclass[12pt]{article}\n\\usepackage{booktabs}\n\\usepackage{geometry}\n\\usepackage{extsizes}\n\\geometry{a4paper, landscape, lmargin=4cm, rmargin=2cm, tmargin=1cm, bmargin=1cm}\n\\begin{document}\n""" + table.to_latex()
        mode = "w"
    else:
        latex = table.to_latex() + "\\end{document}"
        mode = "a"

    with open(filename + ".tex", mode) as f:
        f.write(latex)
        f.write("\n")

def stackoverflow_results():
    sof_light_baseline1 = pd.read_table("average_light_TF_fse_results_baseline1.txt", sep=",")
    sof_light_baseline1.set_index('Benchmarks',inplace=True)
    sof_light_baseline1.index = sof_light_baseline1.index.rename('FSE Benchmarks')

    sof_light_baseline2 = pd.read_table("average_light_TF_fse_results_baseline2.txt", sep=",")
    sof_light_baseline2.set_index('Benchmarks',inplace=True)
    sof_light_baseline2.index = sof_light_baseline2.index.rename('FSE Benchmarks')

    sof_original_baseline1 = pd.read_table("average_Original_TF_fse_results_baseline1.txt", sep=",")
    sof_original_baseline1.set_index('Benchmarks', inplace=True)
    sof_original_baseline1.index = sof_original_baseline1.index.rename('FSE Benchmarks')

    sof_original_baseline2 = pd.read_table("average_Original_TF_fse_results_baseline2.txt", sep=",")
    sof_original_baseline2.set_index('Benchmarks', inplace=True)
    sof_original_baseline2.index = sof_original_baseline2.index.rename('FSE Benchmarks')
    
    assert(sorted(sof_light_baseline1.columns) == sorted(sof_original_baseline1.columns))
    assert(len(sof_light_baseline1.values) == len(sof_original_baseline1.values))

    assert(sorted(sof_light_baseline2.columns) == sorted(sof_original_baseline2.columns))
    assert(len(sof_light_baseline2.values) == len(sof_original_baseline2.values))

    result = pd.concat([sof_light_baseline1, sof_original_baseline1, sof_light_baseline2, sof_original_baseline2])
    result.loc["Gain1"] = result.loc['TensorFlow1'] / result.loc['ShapeFlow1']
    result.loc["Gain1"] = result.loc["Gain1"].round()
    
    result.loc["Gain2"] = result.loc['TensorFlow2'] / result.loc['ShapeFlow2']
    result.loc["Gain2"] = result.loc["Gain2"].round()
    
    result = result.T
    cols = result.columns.tolist()
    cols = cols[0:2] + [cols[-2]] + cols[2:4] + [cols[-1]]
    result = result[cols]
    # Replace Timeouts
    result['ShapeFlow1'].replace(to_replace = 5000.000000, value="Timeout", inplace=True)
    result['ShapeFlow2'].replace(to_replace = 5000.000000, value="Timeout", inplace=True)
    result['TensorFlow1'].replace(to_replace = 5000.000000, value="Timeout", inplace=True)
    result['TensorFlow2'].replace(to_replace = 5000.000000, value="Timeout", inplace=True)
    
    # Replace gains in case of Timeouts
    result.loc[(result['TensorFlow1'].eq('Timeout') | result['ShapeFlow1'].eq('Timeout')), 'Gain1'] = 'N/A'
    result.loc[(result['TensorFlow2'].eq('Timeout') | result['ShapeFlow2'].eq('Timeout')), 'Gain2'] = 'N/A'
    create_latex_from_table(latex_file, result, True)


if __name__ == "__main__":
    stackoverflow_results()
    # github_results()

# def github_results():
#     gh_light_baseline1 = pd.read_table("average_light_TF_GH_results_baseline1.txt", sep=",")
#     gh_light_baseline1.set_index('Benchmarks',inplace=True)
#     gh_light_baseline1.index = gh_light_baseline1.index.rename('Github Benchmarks')

#     gh_light_baseline2 = pd.read_table("average_light_TF_GH_results_baseline2.txt", sep=",")
#     gh_light_baseline2.set_index('Benchmarks',inplace=True)
#     gh_light_baseline2.index = gh_light_baseline2.index.rename('Github Benchmarks')

#     gh_original_baseline1 = pd.read_table("average_Original_TF_GH_results_baseline1.txt", sep=",")
#     gh_original_baseline1.set_index('Benchmarks', inplace=True)
#     gh_original_baseline1.index = gh_original_baseline1.index.rename('Github Benchmarks')

#     gh_original_baseline2 = pd.read_table("average_Original_TF_GH_results_baseline2.txt", sep=",")
#     gh_original_baseline2.set_index('Benchmarks', inplace=True)
#     gh_original_baseline2.index = gh_original_baseline2.index.rename('Github Benchmarks')
    
#     assert(sorted(gh_light_baseline1.columns) == sorted(gh_original_baseline1.columns))
#     assert(len(gh_light_baseline1.values) == len(gh_original_baseline1.values))

#     assert(sorted(gh_light_baseline2.columns) == sorted(gh_original_baseline2.columns))
#     assert(len(gh_light_baseline2.values) == len(gh_original_baseline2.values))

#     result = pd.concat([gh_light_baseline1, gh_original_baseline1, gh_light_baseline2, gh_original_baseline2])
#     result.loc["Gain 1"] = result.loc['TensorFlow 1'] / result.loc['ShapeFlow 1']
#     result.loc["Gain 1"] = result.loc["Gain Baseline1"].round()
    
#     result.loc["Gain Baseline2"] = result.loc['TensorFlow Baseline2'] / result.loc['ShapeFlow Baseline2']
#     result.loc["Gain Baseline2"] = result.loc["Gain Baseline2"].round()
    
#     result = result.T
#     cols = result.columns.tolist()
#     cols = cols[0:2] + [cols[-2]] + cols[2:4] + [cols[-1]]
#     result = result[cols]
    
#     # Replace Timeouts
#     result['ShapeFlow Baseline1'].replace(to_replace = 5000.000000, value="Timeout", inplace=True)
#     result['ShapeFlow Baseline2'].replace(to_replace = 5000.000000, value="Timeout", inplace=True)
#     result['TensorFlow Baseline1'].replace(to_replace = 5000.000000, value="Timeout", inplace=True)
#     result['TensorFlow Baseline2'].replace(to_replace = 5000.000000, value="Timeout", inplace=True)
    
#     # Replace gains in case of Timeouts
#     result.loc[result['TensorFlow Baseline1'].eq('Timeout') | result['ShapeFlow Baseline1'].eq('Timeout'), 'Gain Baseline1'] = 'N/A'
#     result.loc[result['TensorFlow Baseline2'].eq('Timeout') | result['ShapeFlow Baseline2'].eq('Timeout'), 'Gain Baseline2'] = 'N/A'
#     # print(result)
#     create_latex_from_table(latex_file, result, False)
