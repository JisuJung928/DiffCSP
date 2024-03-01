import sys
import pickle
import numpy as np
from mpi4py import MPI
from tqdm import tqdm
from glob import glob
from itertools import chain, combinations

from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher


def make_list(cif_list):
    index_list = []
    structure_list = []
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)

    cif_num = len(cif_list)
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    q = cif_num // size
    r = cif_num % size
    begin = rank * q + min(rank, r)
    end = begin + q
    if r > rank:
        end += 1
    for cif in tqdm(cif_list[begin:end]):
        key1 = int(cif.replace(".", "_").split("_")[1])
        structure1 = Structure.from_file(cif)
        if len(index_list) == 0:
            index_list.append([key1])
            structure_list.append([structure1])
            continue
        overlap = np.zeros(len(structure_list))
        for group_index, structure2_list in enumerate(structure_list):
            for structure2 in structure2_list:
                if matcher.fit(structure1, structure2, symmetric=True):
                    overlap[group_index] = 1
                    break
        # merge
        group_index_list = np.nonzero(overlap)[0]
        if group_index_list.size == 1:
            group_index = group_index_list[0]
            index_list[group_index].append(key1)
            structure_list[group_index].append(structure1)
        elif group_index_list.size > 1:
            for group_index in group_index_list[1:]:
                index_list[group_index_list[0]].extend(index_list.pop(group_index))
                structure_list[group_index_list[0]].extend(structure_list.pop(group_index))
        # np.sum(overlap) == 0
        else:
            index_list.append([key1])
            structure_list.append([structure1])

    return index_list, structure_list


def merge_list(unique_index_list, unique_structure_list):
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)

    group_index_list = list(combinations(np.arange(len(unique_index_list)), 2))
    group_num = len(group_index_list)

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    q = group_num // size
    r = group_num % size
    begin = rank * q + min(rank, r)
    end = begin + q
    if r > rank:
        end += 1
    overlap_list = []

    for indices in tqdm(group_index_list[begin:end]):
        overlap = False
        for structure1 in unique_structure_list[indices[0]]:
            for structure2 in unique_structure_list[indices[1]]:
                if matcher.fit(structure1, structure2, symmetric=True):
                    overlap_list.append(set(indices))
                    overlap = True
                    break
            if overlap:
                break


#    group_num = len(unique_index_list)
#
#    rank = MPI.COMM_WORLD.Get_rank()
#    size = MPI.COMM_WORLD.Get_size()
#    q = group_num // size
#    r = group_num % size
#    begin = rank * q + min(rank, r)
#    end = begin + q
#    if r > rank:
#        end += 1
#    overlap_list = []
#
#    for group1 in tqdm(np.arange(group_num)[begin:end]):
#        max_group_num = max([len(indices) for group, indices in enumerate(unique_index_list) if group != group1])
#        for structure1 in unique_structure_list[group1]:
#            for index in range(max_group_num):
#                for group2, structure2_list in enumerate(unique_structure_list):
#                    if (group1 == group2) or ({group1, group2} in overlap_list):
#                        continue
#                    try:
#                        structure2 = structure2_list[index]
#                    except:
#                        continue
#                    if matcher.fit(structure1, structure2, symmetric=True):
#                        overlap_list.append({group1, group2})

    def find(x, parent):
        """Find the root of the set that contains x."""
        if parent[x] != x:
            parent[x] = find(parent[x], parent)
        return parent[x]

    def union(x, y, parent):
        """Merge the sets that contain x and y."""
        rootX = find(x, parent)
        rootY = find(y, parent)
        if rootX != rootY:
            parent[rootY] = rootX

    def merge_sets(sets):
        element_to_set = {}
        parent = {}

        # Initialize each element to be its own parent
        for s in sets:
            for element in s:
                parent[element] = element
                element_to_set[element] = s

        # Merge sets that have common elements
        for s in sets:
            for element in s:
                union(next(iter(s)), element, parent)

        # Collect elements based on their root parent
        disjoint_sets = {}
        for element in parent:
            root = find(element, parent)
            if root not in disjoint_sets:
                disjoint_sets[root] = set()
            disjoint_sets[root].add(element)

        return list(disjoint_sets.values())

    overlap_set_list = MPI.COMM_WORLD.allgather(overlap_list)
    overlap_set_list = list(chain(*overlap_set_list))
    overlap_set_list = merge_sets(overlap_set_list)

    delete_list = []
    for overlap_set in overlap_set_list:
        overlap_list = list(overlap_set)
        for overlap in overlap_list[1:]:
            unique_index_list[overlap_list[0]].extend(unique_index_list[overlap])
            unique_structure_list[overlap_list[0]].extend(unique_structure_list[overlap])
            delete_list.append(overlap)

    for delete in reversed(sorted(delete_list)):
        del unique_index_list[delete]
        del unique_structure_list[delete]


def main():
    # make structure groups
    cif_list = np.array(glob("*.cif"))
    index_list = []
    for cif in cif_list:
        index_list.append(int(cif.replace(".", "_").split("_")[1]))
    index_list = np.array(index_list)
    cif_list = cif_list[np.argsort(index_list)][:int(sys.argv[1])]
    
    index_list, structure_list = make_list(cif_list)

    # allgather groups
    index_list = MPI.COMM_WORLD.allgather(index_list)
    structure_list = MPI.COMM_WORLD.allgather(structure_list)
    unique_index_list = list(chain(*index_list))
    unique_structure_list = list(chain(*structure_list))

    # merge groups
    merge_list(unique_index_list, unique_structure_list)

    if MPI.COMM_WORLD.Get_rank() == 0:
        with open(f"unique_index_list_{int(sys.argv[1])}.pickle", "wb") as fp:
            pickle.dump(unique_index_list, fp)
        with open(f"unique_structure_list_{int(sys.argv[1])}.pickle", "wb") as fp:
            pickle.dump(unique_structure_list, fp)

        print(f"The number of unique structures: {len(unique_index_list)}")


if __name__ == "__main__":
    main()
