import argparse
from pathlib import Path

if __package__:
    from .experiment import run_experiment
else:
    from experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description='Run a standalone FedCM training example.')
    parser.add_argument('--data-dir', type=Path, default=Path('data'),
                        help='directory for the CIFAR-10 download')
    parser.add_argument('--device', default='cpu', help='PyTorch device, such as cpu or cuda')
    parser.add_argument('--synthetic', action='store_true',
                        help='use the small synthetic example without downloading data')
    args = parser.parse_args()
    run_experiment(args.data_dir, args.device, args.synthetic)


if __name__ == '__main__':
    main()
