import os
import subprocess
from tabulate import tabulate
import numpy as np

repeat_times = 5

def run(all_repeats, average_results, baseline, this_python):
    fse_benchmarks = {}
    for _ in range(repeat_times):
        for dirName in os.listdir("."):
            # if "39032277" in dirName:
            if dirName.isdigit():
                # print(dirName, "hello")
                subdir = "./" + dirName
                fileList = os.listdir(subdir)
                def yes_python(i): return ".py" in i
                fileList = list(filter(yes_python, fileList))
                # print('Found directory: %s' % dirName, fileList)
                # dirName = dirName[2:]
                stored_name0 = dirName + "_" + fileList[0]
                stored_name1 = dirName + "_" + fileList[1]

                if not stored_name0 in fse_benchmarks:
                    fse_benchmarks[stored_name0] = []

                if not stored_name1 in fse_benchmarks:
                    fse_benchmarks[stored_name1] = []

                if "original_tf" in this_python and "38399609" in dirName:
                    baseline = 1        # there is no boston_small dataset in normal TensorFlow
                    # print("yes me", dirName, baseline)
                    # assert False

                # print(stored_name0, stored_name1, "seee")

                p1 = subprocess.Popen(['{} {} {}'.format(this_python, fileList[0], baseline)], shell=True, cwd=subdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)                
                p2 = subprocess.Popen(["grep", "Checkpoint"], stdin=p1.stdout, stdout=subprocess.PIPE)
                p1.stdout.close()
                output = p2.communicate()[0].decode('utf-8')
                # print(output)
                this_file = list(filter(None, output.split("\n")))      # remove empty strings
                total_time = 0
                for krk in this_file:
                    number = krk.split()[-1]
                    if number[-1] == ")":       # for UT-7
                        number = number[:-1]
                    total_time += float(number)
                fse_benchmarks[stored_name0].append(total_time)
                print(stored_name0, total_time)

                p1 = subprocess.Popen(['{} {} {}'.format(this_python, fileList[1], baseline)], shell=True, cwd=subdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)                
                p2 = subprocess.Popen(["grep", "Checkpoint"], stdin=p1.stdout, stdout=subprocess.PIPE)
                p1.stdout.close()
                output = p2.communicate()[0].decode('utf-8')
                # print(output)
                this_file = list(filter(None, output.split("\n")))      # remove empty strings
                total_time = 0
                for krk in this_file:
                    number = krk.split()[-1]
                    if number[-1] == ")":       # for UT-7
                        number = number[:-1]
                    total_time += float(number)
                fse_benchmarks[stored_name1].append(total_time)
                print(stored_name1, total_time)

    
    avergaed_results = {}
    for i in fse_benchmarks:
        avergaed_results[i] = [np.mean(fse_benchmarks[i])]
    #print(tabulate(fse_benchmarks, tablefmt="csv", headers=fse_benchmarks.keys()))


    with open(all_repeats, "w") as f:
        f.write(tabulate(fse_benchmarks, tablefmt="csv", headers=fse_benchmarks.keys()))
    with open(average_results, "w") as f:    
        f.write(tabulate(avergaed_results, tablefmt="csv", headers=avergaed_results.keys()))
    
    print(fse_benchmarks)
    # print(avergaed_results)
    
if __name__ == "__main__":
   # light TF baseline 1
    run("time_light_TF_fse_results_baseline1.txt", "average_light_TF_fse_results_baseline1.txt", 1, "~/light_tf/bin/python")
   # light TF baseline 2
    run("time_light_TF_fse_results_baseline2.txt", "average_light_TF_fse_results_baseline2.txt", 2, "~/light_tf/bin/python")
    # # Original TF baseline 1
    run("time_Original_TF_fse_results_baseline1.txt", "average_Original_TF_fse_results_baseline1.txt", 1, "~/original_tf/original_tf/bin/python")
    # # Original TF baseline 2
    run("time_Original_TF_fse_results_baseline2.txt", "average_Original_TF_fse_results_baseline2.txt", 2, "~/original_tf/original_tf/bin/python")

