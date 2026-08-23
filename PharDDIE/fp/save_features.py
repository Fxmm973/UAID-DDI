
from multiprocessing import Pool
import os
import shutil
import sys
from typing import List, Tuple

from tqdm import tqdm
from tap import Tap

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.data.utils import get_smiles
from chemprop.features import get_available_features_generators, get_features_generator, load_features, save_features
from chemprop.utils import makedirs


class Args(Tap):
    data_path: str='./data/drug_smiles.csv'
    smiles_column: str = 'smiles'
    features_generator: str = 'morgan'
    save_path: str='./features/morgan_dataset1.npz'
    save_frequency: int = 10000
    restart: bool = False
    sequential: bool = False

    def configure(self) -> None:
        self.add_argument('--features_generator', choices=get_available_features_generators())


def load_temp(temp_dir: str) -> Tuple[List[List[float]], int]:
    features = []
    temp_num = 0
    temp_path = os.path.join(temp_dir, f'{temp_num}.npz')

    while os.path.exists(temp_path):
        features.extend(load_features(temp_path))
        temp_num += 1
        temp_path = os.path.join(temp_dir, f'{temp_num}.npz')

    return features, temp_num


def generate_and_save_features(args: Args):
    makedirs(args.save_path, isfile=True)

    smiles = get_smiles(path=args.data_path, smiles_columns=args.smiles_column, flatten=True)
    features_generator = get_features_generator(args.features_generator)
    temp_save_dir = args.save_path + '_temp'

    if args.restart:
        if os.path.exists(args.save_path):
            os.remove(args.save_path)
        if os.path.exists(temp_save_dir):
            shutil.rmtree(temp_save_dir)
    else:
        if os.path.exists(args.save_path):
            raise ValueError(f'"{args.save_path}" already exists and args.restart is False.')

        if os.path.exists(temp_save_dir):
            features, temp_num = load_temp(temp_save_dir)

    if not os.path.exists(temp_save_dir):
        makedirs(temp_save_dir)
        features, temp_num = [], 0

    smiles = smiles[len(features):]

    if args.sequential:
        features_map = map(features_generator, smiles)
    else:
        features_map = Pool().imap(features_generator, smiles)

    temp_features = []
    for i, feats in tqdm(enumerate(features_map), total=len(smiles)):
        temp_features.append(feats)

        if (i > 0 and (i + 1) % args.save_frequency == 0) or i == len(smiles) - 1:
            save_features(os.path.join(temp_save_dir, f'{temp_num}.npz'), temp_features)
            features.extend(temp_features)
            temp_features = []
            temp_num += 1

    try:
        save_features(args.save_path, features)

        shutil.rmtree(temp_save_dir)
    except OverflowError:
        print('Features array is too large to save as a single file. Instead keeping features as a directory of files.')


if __name__ == '__main__':
    generate_and_save_features(Args().parse_args())
