import sys, os

abstract_dict = {
    # "tf.Variable": "abst.var",
    # "tf.random_normal": "abst.rand_normal_",
    "np.random.randn": "abst.random_randn",
    # "tf.matmul": "abst.matmul"
    # "import numpy": "# import numpy"
}

f = open(sys.argv[1])
lines = f.readlines()
f.close()
bool_a = False
for i, l in enumerate(lines):
    matches = []
    for key in abstract_dict.keys():
        if key in l:
            matches.append(key)

    for k in matches:
        lines[i] = lines[i].replace(k, abstract_dict[k])
        bool_a = True

    # if not bool_a:
    #     lines[i] = "# " + l

if bool_a:      # Do not create an extra file if not needed
    name_file = str(sys.argv[1])[:-3] + "_abstracted.py"

    w = open(name_file, "w")
    w.write('import start as abst\n')
    for i in lines:
        w.write(i)
    w.close()
else:
    print("NOT CREATED")
    name_file = str(sys.argv[1])

# os.system('open %s'%(name_file))
os.system('python %s'%(name_file))

# exec('import start as our')
# for i in lines:
#     exec(i)
# exec('print(x)')


