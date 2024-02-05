import os
import time
import logging
from importlib_resources import files as resource_files
from collections import OrderedDict
import numpy as np
import skimage.transform as skitransform
import pickle
import json
from tensorflow.keras.models import load_model
from molSimplify.molscontrol.io_tools import obtain_jobinfo, read_geometry_to_mol, get_geo_metrics, get_bond_order, get_gradient, \
    get_mullcharge, kill_job, check_pid, get_ss_del, get_metal_spin_del
from molSimplify.molscontrol.clf_tools import get_layer_outputs, dist_neighbor, get_entropy, find_closest_model

'''
Main class for the on-the-fly job control.
'''


class dft_control:
    '''
    Attribute of the class:
    step_now: current step for an optimization
    self.mode: mode of the dynamic classifier: "terachem": geo metrics + electronic structure as descriptors, stable for
    terachem users, and "geo": only geometry metrics as descriptors, which can be used for all kinds of quantum
    chesmitry software.
    self.mode_allowed: allowed modes.
    self.step_decisions: steps at which the dynamic classifier can make predictions. (Resizing to be implemented).
    self.scrpath: path to the scratch directory.
    self.geofile: filename of the optimization trajectory in a xyz file format. This is the minimum requirement to use
    the dynamic classifier.
    self.bofile: filename of the trajectory for the bond order matrix (for mode = "terachem").
    self.chargefile: filename of the trajectory for the Mulliken charge(for mode = "terachem").
    self.gradfile: filename of the trajectory for the gradient matrix (for mode = "terachem").
    self.mols: a list of mol3D objects.
    self.features: a combinaton of all feaures of {"feature_name1": [var_step0, var_step1, ...]}
    self.features_norm: self.features after normalization.
    self.feature_mat: input for the dynamic classifier. Dim: (1, step_now, number_features).
    self.preditions: a dictionary of predictions (probabilities of success) with the format of {step_now: prediction}
    self.lses: similar as above but for the latent space entropy (LSE).
    self.train_data: Training (data, labels) for the dynamic classifiers.
    self.status: status of a simulation. True for live and false for dead.
    self.modelfile: path to the model file.
    self.models: a dictory of dynamic classifiers with the format of {step_now: model}
    self.normalization_dict: a dictionary for data normalizations. For geometry metrics, we use value/cutoff
    as standarization, and for other descriptors, we use (value-mean_step0)/std_step0 as standardization.
    self.features_dict: a dictionary for features.
    self.avrg_latent_dist_train: averaged 5-neighbor distance in the latent space for the training data.
    self.resize: whether to use resizing of making predictions on steps where we do not train a model.
    self.lse_cutoff: cutoff for the LSE of model confidence on the prediction.
    self.debug: Whether in debug mode. True means testing on a complete set of files with a finished job. False means
    on-the-fly job control.
    self.pid: pid to kill.
    '''

    def __init__(self, mode='terachem',
                 scrpath='./scr/',
                 geofile='optim.xyz',
                 bofile='bond_order.list',
                 chargefile='charge_mull.xls',
                 gradfile='grad.xyz',
                 mullpopfile="mullpop",
                 outfile='terachem.out',
                 modelsfile=False,
                 normfile=False,
                 traindatafile=False,
                 dataname=False,
                 initxyzfile="initgeo.xyz",
                 modelname='conv',
                 normvecname='norm_dict.json',
                 logfile='./molscontrol.log',
                 task=['geo_flag'],
                 lslayer=-4,
                 lse_cutoff=0.3,
                 use_last_frame=True,
                 spinmult=1,
                 resize=False,
                 debug=False,
                 pid=False):
        self.step_now = -1
        self.mode = mode
        self.mode_allowed = ["terachem", "geo", "oxo"]
        self.step_decisions = [2, 5, 10, 15, 20, 30, 40]
        self.scrpath = scrpath
        self.initxyzfile = initxyzfile
        self.geofile = geofile
        self.bofile = bofile
        self.chargefile = chargefile
        self.gradfile = gradfile
        self.mullpopfile = mullpopfile
        self.outfile = outfile
        self.mols = list()
        self.features = OrderedDict()
        self.features_norm = OrderedDict()
        self.feature_mat = list()
        self.preditions = OrderedDict()
        self.lses = OrderedDict()
        self.lslayer = lslayer
        self.train_data = list()
        self.status = True
        self.modelfile = modelsfile
        self.normfile = normfile
        self.traindatafile = traindatafile
        self.modelname = modelname
        self.dataname = dataname
        self.normvecname = normvecname
        self.logfile = logfile
        self.task = task
        self.use_last_frame = use_last_frame
        self.models = dict()
        self.normalization_dict = dict()
        self.lse_cutoff = lse_cutoff
        self.spinmult = spinmult
        self.resize = resize
        self.debug = debug
        self.pid = pid
        self.features_dict = {
            "terachem": {
                0: 'bo_0', 1: 'bo_sv0', 2: 'bo_offsv0', 3: 'bo_sv1', 4: 'bo_offsv1', 5: 'bo_sv2',
                6: 'bo_offsv2', 7: 'bo_sv3', 8: 'bo_offsv3', 9: 'bo_eq_mean', 10: 'bo_ax_mean',
                11: 'grad_0', 12: 'grad_sv0', 13: 'grad_intsv0', 14: 'grad_sv1',
                15: 'grad_intsv1',
                16: 'grad_sv2', 17: 'grad_intsv2', 18: 'grad_maxnorm', 19: 'grad_intmaxnorm',
                20: 'grad_rms', 21: 'grad_eq_mean', 22: 'grad_ax_mean', 23: 'charge_0',
                24: 'charge_eq_mean', 25: 'charge_ax_mean', 26: 'flag_oct',
                27: 'inspect_oct_angle_devi_max', 28: 'inspect_max_del_sig_angle',
                29: 'inspect_dist_del_all', 30: 'inspect_dist_del_eq',
                31: 'inspect_devi_linear_avrg',
                32: 'inspect_devi_linear_max', 33: 'actural_rmsd_max'
            },
            "geo": {
                0: 'flag_oct', 1: 'inspect_oct_angle_devi_max', 2: 'inspect_max_del_sig_angle',
                3: 'inspect_dist_del_all', 4: 'inspect_dist_del_eq',
                5: 'inspect_devi_linear_avrg',
                6: 'inspect_devi_linear_max', 7: 'actural_rmsd_max'
            },
            "oxo": {
                0: 'bo_0', 1: 'bo_sv0', 2: 'bo_offsv0', 3: 'bo_sv1', 4: 'bo_offsv1', 5: 'bo_sv2',
                6: 'bo_offsv2', 7: 'bo_sv3', 8: 'bo_offsv3', 9: 'bo_eq_mean', 10: 'bo_ax_mean',
                11: 'grad_0', 12: 'grad_sv0', 13: 'grad_intsv0', 14: 'grad_sv1', 15: 'grad_intsv1',
                16: 'grad_sv2', 17: 'grad_intsv2', 18: 'grad_maxnorm', 19: 'grad_intmaxnorm',
                20: 'grad_rms', 21: 'grad_eq_mean', 22: 'grad_ax_mean', 23: 'charge_0',
                24: 'charge_eq_mean', 25: 'charge_ax_mean',
                26: "del_ss", 27: "del_metal_spin", 28: "ss_flag", 29: "metal_spin_flag",
            },
            "custom": {}
        }
        self.scale_by_norm = ['del_ss', 'del_metal_spin',
                              "ss_flag", "metal_spin_flag"]
        self.avrg_latent_dist_train = {
            "terachem": {
                2: 6.34, 5: 7.59, 10: 4.83, 15: 5.21,
                20: 5.06, 30: 9.34, 40: 8.70
            },
            "geo": {
                2: 4.08, 5: 5.35, 10: 5.85, 15: 6.10,
                20: 6.40, 30: 10.44, 40: 9.42
            },
            "oxo": {
                2: 20.563481588788495,
                5: 16.839565509481957,
                10: 15.231886708072244,
                15: 18.282258231748944,
                20: 16.216676625831397,
                30: 34.069884702587714,
                40: 30.742078737505988
                },
            "custom": {}
        }
        self.files_track = {
            "terachem": {
                self.geofile: 0, self.bofile: 0, self.gradfile: 0, self.chargefile: 0
            },
            "geo": {
                self.geofile: 0
            },
            "oxo": {
                self.geofile: 0, self.bofile: 0, self.gradfile: 0, self.chargefile: 0,
                self.mullpopfile: 0, self.outfile: 0
            },
            "custom": {}
        }
        self.file_updated = {
            "terachem": {
                self.geofile: False, self.bofile: False,
                self.gradfile: False, self.chargefile: False
            },
            "geo": {
                self.geofile: False
            },
            "oxo": {
                self.geofile: False, self.bofile: False, self.gradfile: False, self.chargefile: False,
                self.mullpopfile: False, self.outfile: False
            },
            "custom": {}
        }
        self.init_mol = read_geometry_to_mol(self.initxyzfile)
        self.job_info = obtain_jobinfo(self.initxyzfile)
        self.initialize_logger()
        self.initialize_features()
        self.load_models()
        self.load_training_data()
        self.load_normalization_vec()
        self.initilize_file_track_dict()

    def initialize_logger(self):
        logging.basicConfig(filename=self.logfile, filemode='w', level=logging.DEBUG,
                            format='%(name)s - %(levelname)s - %(message)s')
        logging.info(
            '----Logger for the dynamic classifier for on-the-fly job control---')
        logging.info('Monitoring job with PID %s' % str(self.pid))
        if not self.pid:
            logging.warning('NO PID is inputed. Cannot do cany control.')

    def initialize_features(self):
        try:
            for idx, fname in list(self.features_dict[self.mode].items()):
                self.features.update({fname: []})
            logging.info('Feature initialized.')
        except Exception:
            logging.error('Feature initialization failed.', exc_info=True)

    def get_file_path(self, filein):
        if ".out" not in filein:
            return self.scrpath + '/' + filein
        else:
            return filein

    def load_models(self):
        if not self.modelfile:
            modelpath = str(resource_files("molSimplify.molscontrol").joinpath(f"models/{self.mode}"))
        else:
            modelpath = self.modelfile
            logging.warning("Using user-specified models from %s." % modelpath)
        try:
            for step in self.step_decisions:
                _model = '/%s_%d.h5' % (self.modelname, step)
                _modelname = modelpath + _model
                logging.info("Loading model: %s ..." %
                             _modelname.split('/')[-1])
                self.models.update({step: load_model(_modelname)})
        except Exception:
            logging.error('Failed at model loading.', exc_info=True)

    def load_training_data(self):
        if not self.traindatafile:
            datapath = resource_files("molSimplify.molscontrol").joinpath(f"data/{self.mode}/train_data.pkl")
        else:
            datapath = self.traindatafile
            logging.warning("Using user-specified models from %s." % datapath)
        try:
            with open(datapath, 'rb') as f:
                _train_data = pickle.load(f)
            for key, val in list(_train_data.items()):
                self.train_data.append(val)
            logging.info("Training data loaded.")
        except Exception:
            logging.error('Failed at training data loading.', exc_info=True)

    def load_normalization_vec(self):
        if not self.normfile:
            normvecpath = resource_files("molSimplify.molscontrol").joinpath(f"normalization_vec/{self.mode}/norm_dict.json")
        else:
            normvecpath = self.normfile
            logging.warning(
                "Using user-specified models from %s." % normvecpath)
        try:
            with open(normvecpath, 'rb') as f:
                self.normalization_dict = json.load(f)
            logging.info('Normalization vectors loaded')
        except:
            logging.error(
                'Failed at normalization vector loading.', exc_info=True)

    def update_features(self):
        dict_combined = {}
        time.sleep(0.5)
        if not self.use_last_frame:
            frame = self.step_now if self.step_now > -1 else 0
        else:
            frame = -1
        frame_ss_ms = self.step_now if self.step_now > -1 else 0
        if self.mode == 'terachem':
            bondorder_dict = get_bond_order(bofile=self.get_file_path(self.bofile), frame=frame,
                                            job_info=self.job_info, num_sv=4)
            dict_combined.update(bondorder_dict)
            gradient_dict = get_gradient(gradfile=self.get_file_path(self.gradfile), frame=frame,
                                         job_info=self.job_info, num_sv=3)
            dict_combined.update(gradient_dict)
            mullcharge_dict = get_mullcharge(chargefile=self.get_file_path(self.chargefile), frame=frame,
                                             job_info=self.job_info)
            dict_combined.update(mullcharge_dict)
            geometrics_dict = get_geo_metrics(init_mol=self.init_mol, frame=frame, job_info=self.job_info,
                                              geofile=self.get_file_path(self.geofile))
            dict_combined.update(geometrics_dict)
        elif self.mode == 'geo':
            geometrics_dict = get_geo_metrics(init_mol=self.init_mol, frame=frame, job_info=self.job_info,
                                              geofile=self.get_file_path(self.geofile))
            dict_combined.update(geometrics_dict)
        elif self.mode == "oxo":
            bondorder_dict = get_bond_order(bofile=self.get_file_path(self.bofile), frame=frame,
                                            job_info=self.job_info, num_sv=4)
            dict_combined.update(bondorder_dict)
            gradient_dict = get_gradient(gradfile=self.get_file_path(self.gradfile), frame=frame,
                                         job_info=self.job_info, num_sv=3)
            dict_combined.update(gradient_dict)
            mullcharge_dict = get_mullcharge(chargefile=self.get_file_path(self.chargefile), frame=frame,
                                             job_info=self.job_info)
            dict_combined.update(mullcharge_dict)
            dict_combined.update(get_ss_del(self.get_file_path(
                self.outfile), frame=frame_ss_ms))
            dict_combined.update(
                {"ss_flag": 1 if dict_combined["del_ss"] < 1 else 0})
            dict_combined.update(get_metal_spin_del(self.get_file_path(self.mullpopfile), self.spinmult,
                                                    frame=frame_ss_ms,
                                                    idx=self.job_info["natoms"]))
            dict_combined.update(
                {"metal_spin_flag": 1 if dict_combined["del_metal_spin"] < 1 else 0})
        elif self.mode == 'custom':
            if bool(self.features_dict[self.mode]):
                # Placeholder for your functions of obtaining descriptors.
                pass
        else:
            raise KeyError("Mode is not recognized.")
        for idx, fname in list(self.features_dict[self.mode].items()):
            self.features[fname].append(dict_combined[fname])
        with open("features.json", "w") as f:
            json.dump(self.features, f)

    def normalize_features(self):
        for idx, fname in list(self.features_dict[self.mode].items()):
            if fname in list(self.features_dict["geo"].values()) + self.scale_by_norm:
                self.features_norm[fname] = np.array(
                    self.features[fname]) / self.normalization_dict[fname]
            else:
                self.features_norm[fname] = (np.array(self.features[fname]) - self.normalization_dict[fname][0]) / \
                    self.normalization_dict[fname][1]

    def prepare_feature_mat(self):
        self.feature_mat = list()
        for fname, vec in list(self.features_norm.items()):
            self.feature_mat.append(vec)
        self.feature_mat = np.transpose(self.feature_mat)
        self.feature_mat = self.feature_mat.reshape(
            1, self.step_now + 1, len(list(self.features.keys())))

    def resize_feature_mat(self):
        step_chosen, mindelta = find_closest_model(
            self.step_now, self.step_decisions)
        # self.feature_mat = scipy.misc.imresize(self.feature_mat[0], (step_chosen + 1, self.feature_mat.shape[-1]),
        #                                        mode='F', interp='nearest')
        self.feature_mat = skitransform.resize(self.feature_mat,
                                               output_shape=(
                                                   1, step_chosen + 1, self.feature_mat.shape[-1]),
                                               order=3)
        # self.feature_mat = self.feature_mat.reshape(
        #     1, step_chosen + 1, self.feature_mat.shape[-1])
        return step_chosen

    def predict(self, step=False):
        step = self.step_now if not step else step
        pred = self.models[step].predict(self.feature_mat)
        pp = {}
        if len(self.task) == 1:
            pred = [pred]
        for ii in range(len(self.task)):
            pp[self.task[ii]] = pred[ii][0][0]
        self.preditions.update({self.step_now: pp})
        logging.info('At step %d.' % self.step_now)
        logging.info('Predicts: %s ' % str(pp))

    def calculate_lse(self, step=False):
        step = self.step_now if not step else step
        if self.step_now not in self.step_decisions:
            fmat_train = []
            for ii in range(len(self.train_data[0])):
                # fmat_train.append(scipy.misc.imresize(self.train_data[0][ii, :self.step_now + 1, :],
                #                                       (step + 1,
                #                                        self.train_data[0].shape[-1]),
                #                                       mode='F', interp='nearest'))
                fmat_train.append(skitransform.resize(
                    self.train_data[0][ii, :self.step_now + 1, :],
                    output_shape=(step + 1, self.train_data[0].shape[-1]),
                    order=3))
            fmat_train = np.array(fmat_train)
        else:
            fmat_train = self.train_data[0][:, :step + 1, :]
        train_latent = get_layer_outputs(
            self.models[step], self.lslayer, fmat_train, training_flag=False)
        test_latent = get_layer_outputs(
            self.models[step], self.lslayer, self.feature_mat, training_flag=False)
        ll = {}
        for ii in range(len(self.task)):
            __, nn_dists, nn_labels = dist_neighbor(test_latent, train_latent, self.train_data[1][ii],
                                                    l=100, dist_ref=self.avrg_latent_dist_train[self.mode][step])
            lse = get_entropy(nn_dists, nn_labels)
            ll[self.task[ii]] = lse[0]
        self.lses.update({self.step_now: ll})
        logging.info('LSE: %s ' % str(ll))
        logging.info('Prediction summary, %s' % str(self.preditions))
        logging.info('LSE summary, %s' % str(self.lses))

    def make_decision(self):
        killed = False
        for ii, t in enumerate(self.task):
            if self.preditions[self.step_now][t] <= 0.5 and self.lses[self.step_now][t] < self.lse_cutoff:
                logging.critical(
                    "!!!!job killed at step %d!!!!!!" % self.step_now)
                logging.info("Killed for the task of: %s" % t)
                logging.info("Reasons: a prediction of %.4f with LSE of %.4f" % (self.preditions[self.step_now][t],
                                                                                 self.lses[self.step_now][t]))
                self.status = False
                killed = True
        if not killed:
            logging.info(
                'This calculation seems good for now at step %d' % self.step_now)
        else:
            kill_job(self.pid)

    def initilize_file_track_dict(self):
        existed = False
        logging.info(
            "Initialize the files to be tracked during the geometry optimization.")
        logging.info(
            "This may take a while until the first step of SCF calculation finishes...")
        if not bool(self.files_track[self.mode]):
            raise KeyError(
                "file_track is an empty dictory. It should at least contain one file to track on.")
        while not existed:
            for filename in self.files_track[self.mode]:
                filepath = self.get_file_path(filename)
                if os.path.isfile(filepath):
                    self.file_updated[self.mode].update({filename: True})
            existed = all(value is True for value in list(
                self.file_updated[self.mode].values()))
        for filename in self.files_track[self.mode]:
            filepath = self.get_file_path(filename)
            self.files_track[self.mode].update(
                {filename: os.path.getmtime(filepath)})
            self.file_updated[self.mode].update({filename: False})
        logging.info("Tracking files initialization completes.")
        time.sleep(3)
        # Gather features for the 0th step of the optimization
        try:
            self.update_features()
            self.normalize_features()
            self.step_now += 1
            self.prepare_feature_mat()
            logging.info("%d step feature obtained." % self.step_now)
        except Exception:
            logging.warning(
                'Cannot obtain the information of the zeroth step.', exc_info=True)

    def check_updates(self):
        for filename, val in list(self.files_track[self.mode].items()):
            filepath = self.get_file_path(filename)
            if os.path.getmtime(filepath) - val > 3:
                self.file_updated[self.mode].update({filename: True})
        updated = all(value is True for value in list(
            self.file_updated[self.mode].values()))
        if self.debug:
            updated = True
        if updated:
            for filename in self.files_track[self.mode]:
                filepath = self.get_file_path(filename)
                self.files_track[self.mode].update(
                    {filename: os.path.getmtime(filepath)})
                self.file_updated[self.mode].update({filename: False})
            self.step_now += 1
        else:
            for filename, val in list(self.files_track[self.mode].items()):
                self.file_updated[self.mode].update({filename: False})
        return updated

    def update_and_predict(self):
        stop = False
        updated = self.check_updates()
        if self.debug:
            updated = True
        if self.status:
            if updated:
                self.update_features()
                self.normalize_features()
                self.prepare_feature_mat()
                if self.step_now in self.step_decisions:
                    self.predict()
                    self.calculate_lse()
                    self.make_decision()
                elif self.resize and self.step_now > 5:
                    step_chosen = self.resize_feature_mat()
                    logging.info("At step %d, Resizing to activate closet model (%d -> %d)." % (
                        self.step_now, self.step_now, step_chosen))
                    self.predict(step=step_chosen)
                    self.calculate_lse(step=step_chosen)
                    self.make_decision()
                else:
                    logging.info(
                        "At step %d, decision is not activated." % self.step_now)
            if (self.step_now > max(self.step_decisions) and not self.resize) or self.step_now > 80:
                logging.warning(
                    "Step number is larger than the maximum step number that we can make decision (%d steps). The dynamic classifier is now deactivated." % max(
                        self.step_decisions))
                stop = True
        else:
            stop = True
        if self.pid:
            dft_running = check_pid(self.pid)
        else:
            dft_running = True
        if not dft_running:
            stop = True
            logging.info(
                "At step %d, the DFT simulation finishes. molscontrol thus quits." % self.step_now)
        return stop
