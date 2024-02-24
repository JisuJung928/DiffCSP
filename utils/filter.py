import pickle
from mpi4py import MPI
from tqdm import tqdm
from glob import glob
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher


def main():
    comm = MPI.COMM_WORLD
    unique_index_list = {}
    unique_structure_list = {}
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)

    cif_list = sorted(glob("*.cif"))
    cif_num = len(cif_list)
    rank = comm.Get_rank()
    size = comm.Get_size()
    q = cif_num // size
    r = cif_num % size
    begin = rank * q + min(rank, r)
    end = begin + q
    if r > rank:
        end += 1

    # local compare
    for cif in tqdm(cif_list[begin:end]):
        index = cif.replace(".", "_").split("_")[1]
        structure1 = Structure.from_file(cif)
        if len(unique_index_list) == 0:
            unique_index_list[index] = [index]
            unique_structure_list[index] = [structure1]
            continue
        unique = True
        for key, structure2 in unique_structure_list.items():
            if matcher.fit(structure1, structure2[0]):
                unique_index_list[key].append(index)
                unique_structure_list[key].append(structure1)
                unique = False
                break
        if unique:
            unique_index_list[index] = [index]
            unique_structure_list[index] = [structure1]

    unique_index_list = comm.gather(unique_index_list, root=0)
    unique_structure_list = comm.gather(unique_structure_list, root=0)

    # global compare
    if rank == 0:
        for local_unique_index_list, local_unique_structure_list in tqdm(
            zip(unique_index_list[1:], unique_structure_list[1:])
        ):
            for key1, structure1 in local_unique_structure_list.items():
                unique = True
                for key2, structure2 in unique_structure_list[0].items():
                    if matcher.fit(structure1[0], structure2[0]):
                        unique_index_list[0][key2].extend(local_unique_index_list[key1])
                        unique_structure_list[0][key2].extend(structure1)
                        unique = False
                        break
                if unique:
                    unique_index_list[0][key1] = local_unique_index_list[key1]
                    unique_structure_list[0][key1] = structure1

        with open("unique_index.pickle", "wb") as fp:
            pickle.dump(unique_index_list[0], fp)
        with open("unique_structure.pickle", "wb") as fp:
            pickle.dump(unique_structure_list[0], fp)

        print(f"The number of unique structures: {len(unique_index_list[0])}")


if __name__ == "__main__":
    main()
