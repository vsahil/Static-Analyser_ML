import sys, os

abstract_dict = {
    "tf.Variable": "our.var",
    "tf.random_normal": "our.rand_normal_"
}

f = open(sys.argv[1])
lines = f.readlines()

for i, l in enumerate(lines):
    bool_a = False
    for key in abstract_dict.keys():
        if key in l:
            lines[i] = l.replace(key, abstract_dict[key])
            bool_a = True
    if not bool_a:
        lines[i] = "# " + l

exec('import start as our')
for i in lines:
    exec(i)
exec('print(x)')
