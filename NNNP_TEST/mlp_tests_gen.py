import subprocess
import itertools
import argparse
import json
import platform

file_ext = '.bat'

def main(args, test_args):
    # Cmd gen
    arg_dataset = ['--dataset ' + dataset for dataset in test_args['datasets']]
    arg_testratio = ['--testratio ' + str(ratio)
                     for ratio in test_args['testratio']]
    arg_lr = ['--lr ' + str(lr) for lr in test_args['learning_rates']]
    arg_opt = ['--optimizer ' + opt for opt in test_args['optimizers']]
    arg_ep = ['--epochs ' + str(ep) for ep in test_args['epochs']]
    arg_hdims = ['--hdims ' + ' '.join([str(i) for i in hd])
                 for hd in test_args['hdims']]
    arg_coeff = ['--layer-coeff ' + ' '.join([str(i) for i in hd])
                 for hd in test_args['layer_coeff']]

    arglines = itertools.product(
        arg_dataset, arg_testratio, arg_lr, arg_opt, arg_ep, arg_hdims)

    cmdlines = [' '.join([args.python_path, args.test_script]) +
                ' ' + ' '.join(argline) for argline in arglines]

    cmds = []
    for i in range(len(cmdlines)):
        if 'layer' in cmdlines[i]:
            for j in arg_coeff:
                cmds.append(cmdlines[i] + ' ' + j)
        else:
            cmds.append(cmdlines[i])

    if args.direct_run:
        for cmd in cmds:
            subprocess.call(cmd, shell=True)
    else:
        with open('mlp_test'+file_ext, 'w') as script:
            for cmd in cmds:
                script.write(cmd+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-R', '--run', dest='direct_run', help='Run all test cases directly, rather than generating the batch file',
                        default=False, action='store_true')
    parser.add_argument('--pythonpath', dest='python_path', help='Set the path of Python executable',
                        default=None, type=str)
    parser.add_argument('--testscript', dest='test_script', help='Set the path of test script',
                        default='mlp_tester.py', type=str)
    parser.add_argument('--config', dest='config', help='Set the path of JSON test config file',
                        default='config.json', type=str)

    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        json_args = json.load(config_file)
    
    if args.python_path is not None:
        if platform.system() != 'Windows':
            args.python_path = 'python3'
            file_ext = '.sh'
    else:
        args.python_path = 'python'
    
    main(args, json_args)
