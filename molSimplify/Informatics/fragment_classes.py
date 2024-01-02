### This part is dependent on RDKit. Requires RDkit dependency. Uncomment this section
# from rdkit import Chem
# from rdkit import RDLogger
# from rdkit.Chem.rdMolDescriptors import CalcMolFormula
# RDLogger.DisableLog('rdApp.*')

class fragment:
    '''
    This class takes in a name of a fragment
    and stores in the different possibilities
    for that fragment. This fragment then builds macrocycles
    '''
    def __init__(self, name, options, start_macrocycle=False, ring_closure_ind=1):
        '''
        Name is the name (i.e. pyrrole)
        Options is a list of tuples that contain
        the smiles string and metal coordination atom
        The zeroeth element of the options list is what
        will be picked if another possibility is not needed.
        '''
        self.name = name
        self.options = options
        if ring_closure_ind != 1:
            new_options = []
            for smiles_tuple in self.options:
                new_smiles_tuple = self.adjust_ring_closure_index(
                     ring_closure_ind, smiles_tuple)
                new_options.append(new_smiles_tuple)
            self.options = new_options
        if ((start_macrocycle is True)):
            self.start_macrocycle = 9
            new_options = []
            for option in self.options:
                smiles, connection_atom, func_positions = option[0], option[1], option[2]
                new_smiles = self.start_macrocycle_ring(
                    smiles, 0, self.start_macrocycle)
                new_options.append((new_smiles, connection_atom, func_positions))
            self.options = new_options

    def give_compatible_fragments(self, bridge, next_fragment):
        '''
        bridge is the bridging class
        the next fragment is the fragment class for the coming fragment
        '''
        acceptable_fragments = []
        acceptable_second_fragments = []
        acceptable_bridges = []
        possible_bridges = bridge.get_possible_motifs()
        if any(['=' in val[0] for val in possible_bridges]):
            first_fragment_options = self.options
            second_fragment_options = next_fragment.options
        else:
            first_fragment_options = [self.options[0]]
            second_fragment_options = [next_fragment.options[0]]
        for fragment_val in first_fragment_options:
            for second_fragment_val in second_fragment_options:
                for bridge_val in possible_bridges:
                    left, center, right = self.split_smiles_at_lc_adjacent(
                        fragment_val)
                    left2, center2, right2 = self.split_smiles_at_lc_adjacent(
                        second_fragment_val)
                    test_fragment = left + center + \
                        '(' + bridge_val[0] + left2 + center2 + right2 + ')' + right
                    if ('=' in bridge_val[0]):
                        if (not ((bridge_val[0][0] == '=') or ((len(right) > 0) and (right[0] == '=')) or (center[-2] == '='))):
                            continue
                    # ignore the open ring closure for validity
                    test_fragment = test_fragment.replace('9', '')
                    m = Chem.MolFromSmiles(test_fragment)
                    if m is not None:
                        acceptable_bridges.append(bridge_val)
                        acceptable_fragments.append(fragment_val)
                        acceptable_second_fragments.append(second_fragment_val)
        return_val = list(
            set(zip(acceptable_bridges, acceptable_fragments, acceptable_second_fragments)))
        return return_val

    def adjust_ring_closure_index(self, new_index, smiles_tuple=False):
        '''
        This function only works on ring closures with one cycle.
        It identifies the current ring closure and adjusts the index of
        that ring closure based on the user input. If the smiles
        does not actually have a ring in it, then it returns the
        original smiles tuple.
        '''
        if not smiles_tuple:
            new_options = []
            for smiles_tuple in self.options:
                new_smiles_tuple = self.adjust_ring_closure_index(
                     ring_closure_ind, smiles_tuple)
                new_options.append(new_smiles_tuple)
            self.options = new_options
        else:
            smiles, coordination_atom_idx, func_positions = (smiles_tuple[
                0], smiles_tuple[1], smiles_tuple[2])
            digit_indices = [i for i, val in enumerate(smiles) if val.isdigit()]
            new_smiles = ''
            if len(digit_indices) > 2:
                raise ValueError(
                    'This ring closure function should only handle cases with one ring closure. You have more!')
            else:
                for i, val in enumerate(smiles):
                    if i not in digit_indices:
                        new_smiles += val
                    else:
                        new_smiles += str(new_index)
            new_smiles_tuple = (new_smiles, coordination_atom_idx, func_positions)
        return new_smiles_tuple

    def start_macrocycle_ring(self, smiles, start_position=0, macro_ind=9):
        '''
        This function takes a smiles string for monodentate portions and
        returns an adjusted smiles that incorporates the start of a macrocycle.
        The default macrocycle ring index will be 9. If c1ncccc1 is handed in,
        the returned smiles will be c19ncccc1. It simply starts the ring
        opening of the macrocycle.
        '''
        digit_indices = [i for i, val in enumerate(smiles) if val.isdigit()]
        alpha_indices = [i for i, val in enumerate(smiles) if val.isalpha()]
        new_smiles = ''
        started = False
        if len(digit_indices) > 2:
            raise ValueError(
                'We need a monodentate ligand to start the macrocycle. More than one ring closure present.')
        elif len(digit_indices) == 0:
            for i, val in enumerate(smiles):
                if (i == start_position) and (not started):
                    new_smiles += val + str(macro_ind)
                    started = True
                else:
                    new_smiles += val
        else:
            for i, val in enumerate(smiles):
                if (i in digit_indices) and (start_position == 0):
                    if not started:
                        new_smiles += val + str(macro_ind)
                        started = True
                    else:
                        new_smiles += val
                elif (start_position != 0) and (i == alpha_indices[start_position]):
                    new_smiles += val + str(macro_ind)
                    started = True
                else:
                    new_smiles += val
        return new_smiles

    def check_allowed(self, left, center, right):
        forbidden_end = ['=', '/', '\\', ')', '[']
        if left[-1] in forbidden_end:
            center = left[-1] + center
            left = left[:-1]
        if center[-1] in forbidden_end:
            right = center[-1] + right
            center = center[:-1]
        return left, center, right

    def split_smiles_at_lc_adjacent(self, smiles_tuple):
        '''
        This function takes a smiles string and splits it into its respective
        parts that can be functionalized. We allow functionalizations at the
        direct neighbors of the coordinating atom. Thus, this function returns
        a list of substrings that make up the smiles string, so that we can
        stitch together different substructures.

        Takes in a smiles tuple and splits the string into three parts:
        1) between the left adjacent atom and the connection atom
        2) the connection atom itself
        3) between the connection atom and right adjacent atom
        If the coordination atom is the zeroeth index, then by definition, the
        string between the left adjacent atom and the connection atom is going
        to be a blank string. In a similar fashion, if the coordination atom is
        the final index, then by definition, the string between the right adjacent
        atom and the connection atom is going to be a blank string.
        '''
        smiles, coordination_atom_idx = (smiles_tuple[
            0], smiles_tuple[1])
        '''
        The coordination atom index ignores everything except for the letters.
        Thus, we first need to build the indices of letters. We then can get
        the substrings that are constructed by these letters.
        '''
        alphabet_indices = [i for i, val in enumerate(smiles) if val.isalpha()]
        forbidden_end = ['=', '/', '\\', ')']
        if len(alphabet_indices) > 3:
            if (coordination_atom_idx == 0):
                left = smiles[alphabet_indices[0]:alphabet_indices[2]]
                center = smiles[alphabet_indices[2]:]
                right = ''
            elif (coordination_atom_idx == 1):
                left = smiles[alphabet_indices[0]:alphabet_indices[1]]
                center = smiles[alphabet_indices[1]:alphabet_indices[3]]
                right = smiles[alphabet_indices[3]:]
            elif coordination_atom_idx == (len(alphabet_indices) - 1):
                left = smiles[alphabet_indices[0]:alphabet_indices[1]]
                center = smiles[alphabet_indices[1]:alphabet_indices[-1]]
                right = smiles[alphabet_indices[-1]:]
            else:
                left = smiles[0:alphabet_indices[coordination_atom_idx]]
                center = smiles[alphabet_indices[
                    coordination_atom_idx]:alphabet_indices[coordination_atom_idx + 2]]
                right = smiles[alphabet_indices[coordination_atom_idx + 2]:]
        else:
            if (coordination_atom_idx == 0):
                left = smiles[alphabet_indices[0]:alphabet_indices[2]]
                center = smiles[alphabet_indices[2]:]
                right = ''
            else:  # (coordination_atom_idx == 1):
                left = smiles[alphabet_indices[0]:alphabet_indices[1]]
                center = smiles[alphabet_indices[1]:]
                right = ''
        left, center, right = self.check_allowed(left, center, right)
        return left, center, right


class bridge:
    '''
    The bridge class is what is used to determine
    the bonding of the bridging atoms.
    '''
    def __init__(self, name, bonding, func=()):
        '''
        Name is the name (i.e. O) of the bridging atom
        Bonding is a single tuple that defines how the bridging
        atom bonds. For O, it should be (1, 1), which suggests
        two single bonds can be made. For N, it can either be
        (1, 1) which will construct an NH moiety, or (2, 1)
        which will construct an =N- moiety.
        '''
        if name.lower() == 'x':
            name = ''
        self.name = name
        self.bonding = bonding
        self.func = func

    def get_possible_motifs(self):
        if self.bonding == (1, 1):
            return [(self.name, (self.func,))]
        elif (self.bonding == (2, 1)) or (self.bonding == (1, 2)):
            return [(self.name + '=', (self.func,)), ('=' + self.name, (self.func,))]

    def return_name(self):
        if (self.name.lower() in ['x', '']):
            return 'none'
        else:
            if self.bonding == (2, 1) or self.bonding == (1, 2):
                return self.name + '='
            else:
                return self.name


class tetradentate:
    def __init__(self, fragment1, fragment2, fragment3, fragment4, bridge1, bridge2, bridge3):
        '''
        The class takes in four fragments, (1 through 4) that are stitched together
        with bridges. Bridge1 connects fragment1 and fragment2, bridge3 connects fragment2
        and fragment3 as well as fragment4 and fragment1, and bridge2 connects fragment 3
        and fragment4. Only three distinct bridges are allowed to provide
        symmetry constraints. This can be generalized to some arbitrary size.
        '''
        self.fragment1 = fragment1
        self.fragment2 = fragment2
        self.fragment3 = fragment3
        self.fragment4 = fragment4
        self.bridge1 = bridge1
        self.bridge2 = bridge2
        self.bridge3 = bridge3
        self.name = (fragment1.name+'_'+bridge1.return_name()+'_' +
                     fragment2.name+'_'+bridge3.return_name()+'_' +
                     fragment3.name+'_'+bridge2.return_name()+'_' +
                     fragment4.name+'_'+bridge3.return_name())

    def cyclic_equiv(self, u, v):
        n, i, j = len(u), 0, 0
        if n != len(v):
            return False
        while i < n and j < n:
            k = 1
            while k <= n and u[(i + k) % n] == v[(j + k) % n]:
                k += 1
            if k > n:
                return True
            if u[(i + k) % n] > v[(j + k) % n]:
                i += k
            else:
                j += k
        return False

    def check_allowed(self, left, center, right):
        forbidden_end = ['=', '/', '\\', ')', '[']
        if left[-1] in forbidden_end:
            center = left[-1] + center
            left = left[:-1]
        if center[-1] in forbidden_end:
            right = center[-1] + right
            center = center[:-1]
        return left, center, right

    def split_smiles_at_lc_adjacent(self, smiles_tuple):
        '''
        This function takes a smiles string and splits it into its respective
        parts that can be functionalized. We allow functionalizations at the
        direct neighbors of the coordinating atom. Thus, this function returns
        a list of substrings that make up the smiles string, so that we can
        stitch together different substructures.

        Takes in a smiles tuple and splits the string into three parts:
        1) between the left adjacent atom and the connection atom
        2) the connection atom itself
        3) between the connection atom and right adjacent atom
        If the coordination atom is the zeroeth index, then by definition, the
        string between the left adjacent atom and the connection atom is going
        to be a blank string. In a similar fashion, if the coordination atom is
        the final index, then by definition, the string between the right adjacent
        atom and the connection atom is going to be a blank string.
        '''
        smiles, coordination_atom_idx = (smiles_tuple[
            0], smiles_tuple[1])
        '''
        The coordination atom index ignores everything except for the letters.
        Thus, we first need to build the indices of letters. We then can get
        the substrings that are constructed by these letters.
        '''
        alphabet_indices = [i for i, val in enumerate(smiles) if val.isalpha()]
        forbidden_end = ['=', '/', '\\', ')']
        if len(alphabet_indices) > 3:
            if (coordination_atom_idx == 0):
                left = smiles[alphabet_indices[0]:alphabet_indices[2]]
                center = smiles[alphabet_indices[2]:]
                right = ''
            elif (coordination_atom_idx == 1):
                left = smiles[alphabet_indices[0]:alphabet_indices[1]]
                center = smiles[alphabet_indices[1]:alphabet_indices[3]]
                right = smiles[alphabet_indices[3]:]
            elif coordination_atom_idx == (len(alphabet_indices) - 1):
                left = smiles[alphabet_indices[0]:alphabet_indices[1]]
                center = smiles[alphabet_indices[1]:alphabet_indices[-1]]
                right = smiles[alphabet_indices[-1]:]
            else:
                left = smiles[0:alphabet_indices[coordination_atom_idx]]
                center = smiles[alphabet_indices[
                    coordination_atom_idx]:alphabet_indices[coordination_atom_idx + 2]]
                right = smiles[alphabet_indices[coordination_atom_idx + 2]:]
        else:
            if (coordination_atom_idx == 0):
                left = smiles[alphabet_indices[0]:alphabet_indices[2]]
                center = smiles[alphabet_indices[2]:]
                right = ''
            else:  # (coordination_atom_idx == 1):
                left = smiles[alphabet_indices[0]:alphabet_indices[1]]
                center = smiles[alphabet_indices[1]:]
                right = ''
        left, center, right = self.check_allowed(left, center, right)
        return left, center, right

    def sp3_hybridization_checker(self, left, prev_center, right, bridge):
        '''
        This hybridization checker is modeled off of the following:
        full_macrocycle = (left1 + center1 + '(' + frag1[0][0] + left2 + center2 +
                                               '(' + frag2[0][0] + left3 + center3 +
                                               '(' + frag3[0][0] + left4 + center4 +
                                               '(' + frag4[0][0] + str(9) + ')' +
                                               right4 + ')' + right3 + ')' + right2 + ')' + right1)
        This code checks the C hybridization where relevant. Only really relevant
        for planar ligands. Else the code will continue as normal.
        If a ligand is planar (as identified by the presence of a double bond,
        This code then checks if the joining atom is a carbon. If it is a carbon,
        it ensures that the carbon is SP2 hybridized, and not SP3 hybridized. Returns
        true if SP3 hybridized carbon is found.
        '''
        if ('C' in left) and (('=' in left) or ('=' in right)):
            if len(bridge) > 0:
                if (bridge[-1] == '=') or (left[1] == '=') or (len(right) > 1 and right[-2]=='='):
                    returnval = False
                else:
                    returnval = True
            else:
                if (prev_center[-1] == '=') or (left[1] == '=') or (len(right) > 1 and right[-2] == '='):
                    returnval = False
                else:
                    returnval = True
        else:
            # All cases that are not planar (e.g. ethers) will go here.
            returnval = False
        return returnval

    def non_planar(self, left, center, right, frag):
        digits = [val for val in right if val.isdigit()]
        if (len(frag[0][0]) > 0 and len(right) > 0) and ((frag[0][0][0] == '=') and not ((left[0] == '=') or (right[0] == '='))):
            returnval = True
        elif (len(frag[0][0]) == 0) and len(digits) > 0:  # case of zero connection, connection atom must be doubly bonded for planarity
            if '=' not in right[0]:
                returnval = True
            else:
                returnval = False
        else:
            print('INHEA', frag)
            if frag[1][0].count('=') > 0:
                # In this case, we have a planar monodentate fragment. Split the
                # ligand into parts and analyze
                print(frag[1][0], frag[2][0])
                returnval = True
            else:
                returnval = False
        return returnval

    def get_charge(self, smiles):
        pos = smiles.count('+')
        neg = -1*smiles.count('-')
        return neg+pos

    def get_atoms(self, smiles):
        return [val for val in smiles if val.isalpha()]

    def get_formula(self, smiles):
        atoms = self.get_atoms(smiles)
        unique_atoms = set(atoms)
        formula_list = []
        for atom in unique_atoms:
            formula_list.append(atom+str(atoms.count(atom)))
        formula = "".join(formula_list)
        return formula

    def count_atoms(self, smiles):
        return len([val for val in smiles if val.isalpha()])

    def stitch(self):
        frag1_compatible = self.fragment1.give_compatible_fragments(
            self.bridge1, self.fragment2)
        frag2_compatible = self.fragment2.give_compatible_fragments(
            self.bridge3, self.fragment3)
        frag3_compatible = self.fragment3.give_compatible_fragments(
            self.bridge2, self.fragment4)
        frag4_compatible = self.fragment4.give_compatible_fragments(
            self.bridge3, self.fragment1)
        checked = []
        good = []
        checked_canonical = []
        counter = 0
        counta = 0
        flag = False
        results_dict_list = []
        for f1, frag1 in enumerate(frag1_compatible):
            for f2, frag2 in enumerate(frag2_compatible):
                for f3, frag3 in enumerate(frag3_compatible):
                    for f4, frag4 in enumerate(frag4_compatible):
                        check_for_rotation = [
                            frag1[1][0], frag1[0][0], frag2[1][0], frag2[0][0],
                            frag3[1][0], frag3[0][0], frag4[1][0], frag4[0][0]]
                        if any([self.cyclic_equiv(check_for_rotation, val) for val in checked]):
                            continue
                        else:
                            checked.append(check_for_rotation)
                            counta += 1
                        left1, center1, right1 = self.split_smiles_at_lc_adjacent(frag1[1])
                        left2, center2, right2 = self.split_smiles_at_lc_adjacent(frag2[1])
                        left3, center3, right3 = self.split_smiles_at_lc_adjacent(frag3[1])
                        left4, center4, right4 = self.split_smiles_at_lc_adjacent(frag4[1])

                        if len(frag4[0][0]) == 0:
                            # This has to be a separate case because of the ring closure. We don't need parentheses in this case.
                            full_macrocycle = (left1 + center1 + '(' + frag1[0][0] + left2 + center2 +
                                               '(' + frag2[0][0] + left3 + center3 +
                                               '(' + frag3[0][0] + left4 + center4 + frag4[0][0] + str(9) +
                                               right4 + ')' + right3 + ')' + right2 + ')' + right1)
                        else:
                            full_macrocycle = (left1 + center1 + '(' + frag1[0][0] + left2 + center2 +
                                               '(' + frag2[0][0] + left3 + center3 +
                                               '(' + frag3[0][0] + left4 + center4 +
                                               '(' + frag4[0][0] + str(9) + ')' +
                                               right4 + ')' + right3 + ')' + right2 + ')' + right1)

                        if self.sp3_hybridization_checker(left2, center1, right2, frag1[0][0]):
                            continue
                        if self.sp3_hybridization_checker(left3, center2, right3, frag2[0][0]):
                            continue
                        if self.sp3_hybridization_checker(left4, center3, right4, frag3[0][0]):
                            continue
                        if self.sp3_hybridization_checker(left1, center4, right1, frag4[0][0]):
                            continue
                        m = Chem.MolFromSmiles(full_macrocycle)
                        if m is not None:
                            test = Chem.MolToSmiles(m, canonical=True, isomericSmiles=False)
                            charge = self.get_charge(full_macrocycle)
                            if test in checked_canonical:
                                continue
                            else:
                                print(self.fragment1.name, self.fragment3.name,
                                      self.bridge1.return_name(), self.bridge2.return_name(),
                                      self.bridge3.return_name())
                                func_1, func_2, func_3, func_4 = False, False, False, False
                                bridge1_func, bridge2_func, bridge3_func = False, False, False
                                coord_2 = (frag1[1][1] +
                                           self.count_atoms(center1) +
                                           self.count_atoms(frag1[0][0]) +
                                           self.count_atoms(left1))
                                coord_3 = (coord_2 + self.count_atoms(center2) +
                                           self.count_atoms(frag2[0][0]) +
                                           self.count_atoms(left3))
                                coord_4 = (coord_3 + self.count_atoms(center3) +
                                           self.count_atoms(frag3[0][0]) +
                                           self.count_atoms(left4))
                                if len(frag1[1][2]) > 0:  # frag 1 has functionalizable positions
                                    func_1 = [val + self.count_atoms(frag1[0][0] + left2 + center2 +
                                                                     '(' + frag2[0][0] + left3 + center3 +
                                                                     '(' + frag3[0][0] + left4 + center4 +
                                                                     frag4[0][0] + str(9) + right4 + ')' +
                                                                     right3 + ')' + right2)
                                              for val in frag1[1][2]]
                                if len(frag2[1][2]) > 0:
                                    func_2 = [val + self.count_atoms(frag1[0][0] + left2 + center2 +
                                                                     frag2[0][0] + left3 + center3 +
                                                                     '(' + frag3[0][0] + left4 + center4 +
                                                                     frag4[0][0] + str(9) +
                                                                     right4 + ')' + right3)
                                              for val in frag2[1][2]]
                                if len(frag3[1][2]) > 0:
                                    func_3 = [val + self.count_atoms(frag1[0][0] + left2 + center2 +
                                                                     frag2[0][0] + left3 + center3 +
                                                                     frag3[0][0] + left4 + center4 +
                                                                     frag4[0][0] + str(9) + right4)
                                              for val in frag3[1][2]]
                                if len(frag4[1][2]) > 0:
                                    func_4 = [val + self.count_atoms(frag1[0][0] + left2 + center2 +
                                                                     frag2[0][0] + left3 + center3 +
                                                                     frag3[0][0] + left4 + center4 +
                                                                     frag4[0][0] + str(9))
                                              for val in frag4[1][2]]

                                if len(frag1[0][1]) > 0:  # frag 1 has functionalizable positions
                                    bridge1_func = [self.count_atoms(left1 + center1) for _ in frag1[0][1]]
                                if len(frag3[0][1]) > 0:
                                    bridge2_func = [self.count_atoms(left1 + center1 + '(' + frag1[0][0] + left2 + center2 +
                                                    '(' + frag2[0][0] + left3 + center3) for _ in frag3[0][1]]

                                if len(frag2[0][1]) > 0:
                                    # ## bridge 3 is symmetric
                                    bridge3_func_1 = [self.count_atoms(left1 + center1 + '(' + frag1[0][0] + left2 + center2)
                                                      for _ in frag2[0][1]]
                                    bridge3_func_2 = [self.count_atoms(left1 + center1 + '(' + frag1[0][0] + left2 + center2 +
                                                      '(' + frag2[0][0] + left3 + center3 +
                                                      '(' + frag3[0][0] + left4 + center4)for _ in frag2[0][1]]
                                    bridge3_func = [bridge3_func_1, bridge3_func_2]

                                # At the end here we are adding back ones because smicat is 1 indexed
                                # instead of 0
                                coord_atoms_zero_index = [
                                    frag1[1][1], coord_2, coord_3, coord_4]
                                coord_atoms_smicat = [
                                    frag1[1][1] + 1, coord_2 + 1, coord_3 + 1, coord_4 + 1]
                                coord_elements = [self.get_atoms(frag1[1][0])[frag1[1][1]].strip('([/-+])'),
                                                  self.get_atoms(frag2[1][0])[frag2[1][1]].strip('([/-+])'),
                                                  self.get_atoms(frag3[1][0])[frag3[1][1]].strip('([/-+])'),
                                                  self.get_atoms(frag4[1][0])[frag4[1][1]].strip('([/-+])')]
                                temp_dict = {'name': self.name,
                                             'frag1': self.fragment1.name,
                                             'frag2': self.fragment2.name,
                                             'frag3': self.fragment3.name,
                                             'frag4': self.fragment4.name,
                                             'bridge1': self.bridge1.return_name(),
                                             'bridge2': self.bridge2.return_name(),
                                             'bridge3': self.bridge3.return_name(),
                                             'frag1_smiles': frag1[1][0],
                                             'frag2_smiles': frag2[1][0],
                                             'frag3_smiles': frag3[1][0],
                                             'frag4_smiles': frag4[1][0],
                                             'frag1_func': func_1,
                                             'frag2_func': func_2,
                                             'frag3_func': func_3,
                                             'frag4_func': func_4,
                                             'bridge1_func': bridge1_func,
                                             'bridge2_func': bridge2_func,
                                             'bridge3_func': bridge3_func,
                                             'macrocycle_smiles': full_macrocycle,
                                             'coord_atoms_zero_index': coord_atoms_zero_index,
                                             'coord_atoms_smicat': coord_atoms_smicat,
                                             'coord_elements': coord_elements,
                                             'formula': CalcMolFormula(m),
                                             'charge': charge,
                                             'size': self.count_atoms(full_macrocycle),
                                             'canonical_smiles': test
                                             }
                                results_dict_list.append(temp_dict)
                                good.append((full_macrocycle, coord_atoms_smicat, charge))
                                checked_canonical.append(test)
                            counter += 1
        return results_dict_list
