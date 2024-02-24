import sys
from pymatgen.io.cif import CifParser
from pymatgen.io.lammps.data import LammpsData


parser = CifParser(sys.argv[1])
#structure = parser.get_structures()[0]
structure = parser.parse_structures(primitive=True)[0]
lammps_data = LammpsData.from_structure(structure, atom_style='atomic')
lammps_data.write_file(sys.argv[2])
