import os
import subprocess
from tabulate import tabulate
import numpy as np


# def create_latex_from_table(filename: str, table):
#     table = tabulate(table_info, tablefmt="latex", headers=headers)
#     # print(table)
#     latex = "\\documentclass{article}\n\\pdfpageheight=11in\n\\pdfpagewidth=8.5in\n\\begin{document}\n" + table + "\\end{document}"
#     with open(filename + ".tex", 'w') as f:
#         f.write(latex)

# for item in os.listdir("Github/"):
#     for fileList in os.listdir("Github/"+item):
#         print(fileList, "see")
# exit()
repeat_times = 5
GH_benchmarks = {}
for _ in range(repeat_times):
    root = "Github/"
    for item in os.listdir(root):
    # for dirName, subdirList, fileList in os.walk("Github/UT-9"):
        subdir = root + item 
        for fileList in os.listdir(subdir):
            if ".py" in fileList:
                # print(item, "see")
            # if "UT-" in dirName and not "-buggy" in dirName and not "-fix" in dirName and not "_data" in dirName and fileList and not "__pycache__" in dirName and not "logs" in dirName:       # non-empty fileList and not a pycache
                # print('Found directory: %s' % dirName, fileList)
                x = subdir.split("/")[1:]
                stored_name0 = x[0] + "_buggy"
                stored_name1 = x[0] + "_fix"

                if not stored_name0 in GH_benchmarks:
                    GH_benchmarks[stored_name0] = []

                if not stored_name1 in GH_benchmarks:
                    GH_benchmarks[stored_name1] = []

                # count = 0
                # for fname in :
                #     print('\t%s' % fname)
                #     if ".py" in fname:
                #         # print(fname, "SEE")
                #         count += 1
                # assert(count == 1), "only 1 python file expected: {}".format(fileList)
                # for fname in fileList:
                    # if ".py" in fname:
                
                if "UT-1" in subdir or "UT-9" in subdir:
                    this_python = "~/original_tf/original_tf/bin/python"
                    sequence_buggy = 0
                    sequence_fix = 1
                elif "UT-7" in subdir:
                    this_python = "~/TensorFlow-Program-Bugs-master/Github/UT-7/tf_0.12/bin/python"
                    sequence_buggy = 2
                    sequence_fix = 3
                else:
                    assert False, "Got this subdir: {}".format(subdir)

                p1 = subprocess.Popen(['{} {} {}'.format(this_python, fileList, sequence_buggy)], shell=True, cwd=subdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)                
                p2 = subprocess.Popen(["grep", "Checkpoint"], stdin=p1.stdout, stdout=subprocess.PIPE)
                p1.stdout.close()
                output = p2.communicate()[0].decode('utf-8')
                this_file = list(filter(None, output.split("\n")))      # remove empty strings
                total_time = 0
                for krk in this_file:
                    number = krk.split()[-1]
                    if number[-1] == ")":       # for UT-7
                        number = number[:-1]
                    total_time += float(number)
                print(stored_name0, total_time)
                GH_benchmarks[stored_name0].append(total_time)


                p1 = subprocess.Popen(['{} {} {}'.format(this_python, fileList, sequence_fix)], shell=True, cwd=subdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)                
                p2 = subprocess.Popen(["grep", "Checkpoint"], stdin=p1.stdout, stdout=subprocess.PIPE)
                p1.stdout.close()
                output = p2.communicate()[0].decode('utf-8')
                this_file = list(filter(None, output.split("\n")))      # remove empty strings
                total_time = 0
                for krk in this_file:
                    number = krk.split()[-1]
                    if number[-1] == ")":       # for UT-7
                        number = number[:-1]
                    total_time += float(number)
                print(stored_name1, total_time)
                GH_benchmarks[stored_name1].append(total_time)

avergaed_results = {}
for i in GH_benchmarks:
    avergaed_results[i] = [np.mean(GH_benchmarks[i])]
print(tabulate(GH_benchmarks, tablefmt="csv", headers=GH_benchmarks.keys()))

with open("time_original_TF_GH_results.txt", "w") as f:
    f.write(tabulate(GH_benchmarks, tablefmt="csv", headers=GH_benchmarks.keys()))
with open("averaged_original_tf_GH.txt", "w") as f:    
    f.write(tabulate(avergaed_results, tablefmt="csv", headers=avergaed_results.keys()))

