import os
import subprocess

python_path = 'python'
conf_files = []
for f in os.listdir('./test_config/'):
    if f.endswith('.json'):
        conf_files.append(os.path.join('.', 'test_config', f))

cmds = [python_path + f' mlp_tests_gen.py -R --config {i}' for i in conf_files]

for cmd in cmds:
    subprocess.call(cmd, shell=False)
