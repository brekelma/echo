import importlib
import model
import dataset
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='test_config.json')
parser.add_argument('--noise', type=str)
parser.add_argument('--filename', type=str, default='latest_unnamed_run')
parser.add_argument('--beta', type=float)
parser.add_argument('--validate', type=bool, default = 1)
parser.add_argument('--verbose', type=bool, default = 0)
parser.add_argument('--fit_gen', type=bool, default = 1)
parser.add_argument('--per_label')
parser.add_argument('--dataset', type=str, default = 'binary_mnist')
args, _ = parser.parse_known_args()

if ".json" in args.config:
	config = args.config
else:
	config = json.loads(args.config.replace("'", '"'))

if args.dataset == 'fmnist':
        d = dataset.fMNIST()
elif args.dataset == 'binary_fmnist':
        d = dataset.fMNIST(binary = True)
elif args.dataset == 'binary_mnist':
        d = dataset.MNIST(binary= True)
elif args.dataset == 'mnist':
        d = dataset.MNIST()
elif args.dataset in ['omniglot', 'omni']:
        d = dataset.Omniglot()
elif args.dataset == 'dsprites':
        d = dataset.DSprites()
elif args.dataset == "cifar10" or args.dataset == 'cifar':
        d = dataset.Cifar10()

if args.per_label is not None:
        d.shrink_supervised(args.per_label)

m = model.NoiseModel(d, config = config, filename = args.filename, verbose = args.verbose)


if args.noise is not None:
	if args.noise == 'multiplicative':
		m.layers[0]['layer_kwargs']['multiplicative'] = True
	if args.noise == 'additive':
		m.layers[0]['layer_kwargs']['multiplicative'] = False
if args.beta is not None:
	for loss in m.losses:
		if not isinstance(loss, dict): # reconstruction is a Loss object, so will be skipped
			continue
		if 'mmd' in loss['type']:
			pass
		elif loss['weight'] == 0:
			pass
		else:
			loss['weight'] = args.beta


m.fit(d.x_train, verbose = args.verbose, validate = args.validate, fit_gen = args.fit_gen)
