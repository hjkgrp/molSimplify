# @file rundiag.py
#  Contains run_diag class for ANN
#
#  Written by Kulik group
#
#  Department of Chemical Engineering, MIT


class run_diag:
    """Class of run diagnostic information to automated decision making and property prediction
    """

    # Constructor
    #  @param self The object pointer
    def __init__(self):
        self.sanity_is_set = False  # flag to indicate if properties
        # have been written to this file
        self.ANN_is_set = False
        self.bl_is_set = False
        self.mol_is_set = False
        self.catalysis_is_set = False
        self.sanity = False
        self.min_dist = False
        self.ANN_flag = False  # ANN value has been set?
        self.ANN_reason = " not set"  # Reason ANN not set
        self.ANN_attributes = dict()  # placeholder for
        # predicted properties
        self.catalysis_flag = False
        self.catalysis_reason = " not activated"
        self.dict_bondl = False  # stores the ML-dict bond dist
        self.mol = False  # stores a mol3D representation of the mol.

    ########################################
    ### class methods needed to populate ###
    ########################################

    def set_sanity(self, sanity, min_distance):
        """Set the sanity of predictions.

        Parameters
        ----------
            sanity : bool
                Flag for whether the ANN variables exist.
            min_distance : float
                Minimum distance to train
        """
        if not self.sanity_is_set:
            self.sanity_is_set = True
        self.sanity = sanity
        self.min_dist = min_distance

    def set_ANN(self, ANN_flag, ANN_reason=False, ANN_dict=False, catalysis_flag=False, catalysis_reason=False):
        """Set the ANN properties.

        Parameters
        ----------
            ANN_flag : bool
                Flag for whether the ANN variables exist.
            ANN_reason : str, optional
                Reasoning for why ANN failed if failed. Default is False.
            ANN_dict : dict, optional
                Dictionary with ANN values and uncertainty.
            catalysis_flag : bool, optional
                Whether or not catalytic properties are set.
            catalysis_reason : str, optional
                Reasoning for why catalytic ANN failed if failed. Default is False.
        """
        if not self.ANN_is_set:
            self.ANN_is_set = True
        self.ANN_flag = ANN_flag
        if not ANN_flag:
            self.ANN_reason = ANN_reason
        elif ANN_flag:
            self.ANN_attributes = ANN_dict
        if not self.catalysis_is_set:
            self.catalysis_is_set = True
        self.catalysis_flag = catalysis_flag
        if not catalysis_flag:
            self.catalysis_reason = catalysis_reason
        elif catalysis_flag:
            self.ANN_attributes = ANN_dict

    def set_dict_bl(self, dict_bl):
        """Set the ANN properties.

        Parameters
        ----------
            dict_bl : dict, optional
                Dictionary with ANN bond lengths.

        """
        if not self.bl_is_set:
            self.bl_is_set = True
        self.dict_bondl = dict_bl

    def set_mol(self, mol):
        """Set the ANN molecule.

        Parameters
        ----------
            mol : mol3D
                mol3D class instance for optimized molecule.

        """
        if not self.mol_is_set:
            self.mol_is_set = True
        self.mol = mol

    ########################################
    ### class methods needed to report  ####
    ########################################
    def write_report(self, path):
        """Write report of molecule with ANN properties.

        Parameters
        ----------
            path : str
                Path for location to write the report.

        """
        report = []
        if (not self.sanity_is_set) and (not self.ANN_is_set) and (not self.bl_is_set):
            report.append('No diagnostic set')
        else:
            if self.sanity_is_set:
                report.append('Bad structure?, ' + str(self.sanity))
                if not self.sanity:
                    report.append('Min_dist (A), ' + str(self.min_dist))
            if self.ANN_is_set:
                report.append('Was ANN used?, '+str(self.ANN_flag))
                if not self.ANN_flag:
                    report.append('ANN reason, ' + str(self.ANN_reason))
                else:
                    for keys in list(self.ANN_attributes.keys()):
                        report.append(str(keys) + ', ' +
                                      str(self.ANN_attributes[keys]))
            if self.catalysis_is_set:
                report.append('Was Catalytic ANN used?, ' +
                              str(self.catalysis_flag))
                if not self.catalysis_flag:
                    report.append('Catalytic ANN reason, ' +
                                  str(self.catalysis_reason))
                else:
                    for keys in list(self.ANN_attributes.keys()):
                        report.append(str(keys) + ', ' +
                                      str(self.ANN_attributes[keys]))
            if self.bl_is_set:
                report.append('ML-bl (database, A), ' + str(self.dict_bondl))
        with open(path, 'w') as f:
            for lines in report:
                f.write(lines + '\n')
