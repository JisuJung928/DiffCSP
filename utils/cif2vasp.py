import sys
from pymatgen.io.cif import CifParser
from pymatgen.io.vasp.inputs import Poscar


parser = CifParser(sys.argv[1])
#structure = parser.get_structures()[0]
structure = parser.parse_structures(primitive=True)[0]
poscar = Poscar(structure)
poscar.write_file(sys.argv[2])
