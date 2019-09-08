import os
import subprocess
from tabulate import tabulate
import numpy as np


repeat_times = 5
SOF_benchmarks = {}
for _ in range(repeat_times):
    for dirName, subdirList, fileList in os.walk("StackOverflow"):
        if "UT-" in dirName and fileList and not "__pycache__" in dirName:       # non-empty fileList and not a pycache
            print('Found directory: %s' % dirName, fileList)
            x = dirName.split("/")[1:]
            stored_name = x[0] + "_" + x[1].split("-")[-1]
            # try:
            #     assert(not stored_name in SOF_benchmarks), "the dirname should not be already present"
            # except:
            #     print(dirName, stored_name, os.path.basename(dirName))
            #     assert False
            if not stored_name in SOF_benchmarks:
                SOF_benchmarks[stored_name] = []

            count = 0
            for fname in fileList:
                print('\t%s' % fname)
                if ".py" in fname:
                    # print(fname, "SEE")
                    count += 1
            assert(count == 1), "only 1 python file expected: {}".format(fileList)
            for fname in fileList:
                if ".py" in fname:
                    p1 = subprocess.Popen(['python {}'.format(fname)], shell=True, cwd=dirName, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)                
                    p2 = subprocess.Popen(["grep", "Checkpoint"], stdin=p1.stdout, stdout=subprocess.PIPE)
                    p1.stdout.close()
                    output = p2.communicate()[0].decode('utf-8')
                    this_file = list(filter(None, output.split("\n")))      # remove empty strings
                    total_time = 0
                    for krk in this_file:
                        total_time += float(krk.split()[-1])
                    print(stored_name, total_time)
                    SOF_benchmarks[stored_name].append(total_time)
            # exit()
avergaed_results = {}
for i in SOF_benchmarks:
    avergaed_results[i] = [np.mean(SOF_benchmarks[i])]

with open("time_Original_TF_SOF_results.txt", "w") as f:
    f.write(tabulate(SOF_benchmarks, tablefmt="csv", headers=SOF_benchmarks.keys()))
with open("averaged_originaltf_sof.txt", "w") as f:    
    f.write(tabulate(avergaed_results, tablefmt="csv", headers=avergaed_results.keys()))





        # f.write(tabulate(SOF_benchmarks, headers=SOF_benchmarks.keys()))
# print(avergaed_results)
    # with open("Original_TF_SOF_results.txt", "a") as f:
    #     f.write("\nORIGINAL TF STACKOVERFLOW\n")
    #     f.write(tabulate(SOF_benchmarks, headers=SOF_benchmarks.keys()))
