import os
import pandas as pd
from glob import glob
import argparse

from pymatgen.io.vasp import Poscar
from pymatgen.io.cif import CifWriter


def parse_log(filename):
    """
    parse_log reads the last energy as written in OUTCAR.
    """
    with open(filename, "r") as fp:
        for line in fp:
            if 'free  ' in line:
                energy = float(line.split()[-2])
    return energy


def main(args):
    """
    python vasp2csv.py --structure_files poscar_1 poscar_2 ...
                       --log_files outcar_1 outcar_2 ...
    """
    csv = pd.DataFrame(columns=["material_id", "free_energy", "cif"])
    for count, (structure, log) in enumerate(zip(args.structure_files, args.log_files)):
        try:
            structure = Poscar.from_file(structure, check_for_potcar=False).structure
        except Exception:
            print(f"{structure} does not exist\n")
            continue
        energy = parse_log(log)
        cifwriter = CifWriter(structure)
        cifwriter.write_file("tmp.cif")
        with open("tmp.cif", "r") as fp:
            cif = fp.read()
        csv = pd.concat(
            [
                csv,
                pd.DataFrame(
                    data=[[count, energy, cif]],
                    columns=["material_id", "free_energy", "cif"],
                ),
            ],
            ignore_index=True,
        )
        os.remove("tmp.cif")
        print(f"{count + 1} / {len(args.structure_files)}")
    # split data into train(8):valid(1):test(1)
    train = csv.sample(frac=0.8)
    test = csv.drop(train.index)
    valid = test.sample(frac=0.5)
    test = test.drop(valid.index)
    # save it
    train.to_csv("train.csv")
    valid.to_csv("val.csv")
    test.to_csv("test.csv")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--structure_files", nargs="*", help="vasp structure files", required=True
    )
    args.add_argument("--log_files", nargs="*", help="vasp outcar files", required=True)
    args = args.parse_args()
    main(args)
