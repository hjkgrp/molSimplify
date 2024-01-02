import pickle
import os


def try_float(obj):
    """Tries to convert an item into a floating point value.

        Parameters
        ----------
            obj : str, int
                Object of any type to be converted to float.

        Returns
        -------
            floating_point : float
                Floating point version of object. If fails, returns object itself.

    """
    # Converts an object to a floating point if possible
    try:
        floating_point = float(obj)
    except ValueError:
        floating_point = obj
    return floating_point


def strip_new_line(string):
    """Tries to strip string of new line.

        Parameters
        ----------
            string : str
                Input string.

        Returns
        -------
            output : str
                Output string with newline characters removed..

    """
    if string[-1] == '\n':
        return string[:-1]
    else:
        return string


class resub_history:
    """Resub history class that stores the information about a given job.
    Class for saving information about previous resubmissions.

        Parameters
        ----------
            path : str, optional
                Path to place resub history object. Default is None.

        Example use of history object

        >>> resub = resub_history()
        # do this step even if no pickle file exists already
        >>> resub.read(outfile_path)
        >>> resub.save() # update stored values

    """

    def __init__(self, path=None):
        self.resub_number = 0
        self.status = 'Normal'
        self.needs_resub = False
        self.notes = []
        self.outfiles = []
        self.infiles = []
        self.xyzs = []
        self.jobscripts = []
        self.waiting = None  # Path to another job that this job is waiting on
        self.path = path
        self.manually_abandoned = False

    def save(self):
        """Saves the current status of the history object.
        """

        if self.path is None:
            raise Exception(
                'The path for the resub_history pickel file is not specified!')
        with open(self.path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def read(self, path):
        """Read an output file with a given path.

            Parameters
            ----------
                path : str, optional
                    Path of output file to read into history object.

        """

        if path.endswith('.out'):
            path = path.rsplit('.', 1)[0]+'.pickle'

        if os.path.isfile(path):
            with open(path, 'rb') as handle:
                saved = pickle.load(handle)
            self.resub_number = saved.resub_number
            self.status = saved.status
            self.needs_resub = saved.needs_resub
            self.notes = saved.notes
            self.outfiles = saved.outfiles
            if hasattr(saved, 'infiles'):
                self.infiles = saved.infiles
            if hasattr(saved, 'xyzs'):
                self.xyzs = saved.xyzs
            if hasattr(saved, 'jobscripts'):
                self.jobscripts = saved.jobscripts
            if hasattr(saved, 'waiting'):
                self.waiting = saved.waiting
            if hasattr(saved, 'manually_abandoned'):
                self.manually_abandoned = saved.manually_abandoned
        self.path = path

    def abandon(self):
        """Bound method to abandon a job manually.
        """
        self.manually_abandoned = True
        self.status = 'Manually abandoned'


class textfile:
    """
    Class for importing textfiles in a searchable way.
        Parameters
        ----------
            file_name : str, optional
                Path to file to read. Default is None.
    """
    def __init__(self, file_name=None):
        """
        Initilization for textfile

        Parameters
        ----------
            file_name : str, optional
                Path to file to read. Default is None.
        """
        if file_name:
            with open(file_name, 'r',errors='ignore') as fin:
                try:
                    self.lines = fin.readlines()
                except:
                    raise Exception(f'Error reading {file_name}')
            self.lines = [strip_new_line(i) for i in self.lines]

        else:
            self.lines = None

    def wordgrab(self, keywords, indices, last_line=False, first_line=False, min_value=False, matching_index=False):
        """Method to grab words from text. Takes two lists as input.

            Parameters
            ----------
                keywords : list
                    Keywords to look for.
                indices : list
                    Indices to pull from the matching lines
                last_line : bool, optional
                    Gets the last instance of the match in the text file. Default is False.
                first_line : bool, optional
                    Gets the first instance of the match in the text file. Default is False.
                min_value : bool, optional
                    Gets the smallest instance of the match in the text file. Default is False. Numeric values converted to floats.
                matching_index : bool, optional
                    Gives you the indices of the lines that match. Default is False.

            Returns
            -------
                results_to_return : list
                    List of results of scraping. Be careful with the nesting (returns a list of lists).

        """
        # takes two lists as an input
        # The first list is the keywords to look for
        # The second list is the indices to pull from the matching lines
        #  Returns a list of the resulting values. Numeric values are automatically converted to floats
        # Function is most intuitive when ONE of the following is set to True (last_line,first_line,min_value)
        #     Otherwise, the list will be nested one more time than you may expect

        if type(keywords) != list:
            keywords = [keywords]
        if type(indices) != list:
            indices = [indices]*len(keywords)

        results = dict()
        zipped_values = list(zip(keywords, indices, list(range(len(keywords)))))

        for counter, line in enumerate(self.lines):
            for keyword, index, keyword_number in zipped_values:
                if keyword in line:

                    if type(index) == int:
                        matching_value = try_float(line.split()[index])
                    else:
                        matching_value = line.split()

                    # Normal Procedure
                    if not matching_index:
                        if keyword_number not in list(results.keys()):
                            results[keyword_number] = [matching_value]
                        else:
                            results[keyword_number].append(matching_value)

                    # Special procedure for returning the index of matching lines instead of the matching values
                    if matching_index:
                        if keyword_number not in list(results.keys()):
                            results[keyword_number] = [counter]
                        else:
                            results[keyword_number].append(counter)

        if (last_line and min_value) or (last_line and first_line) or (first_line and min_value):
            raise ValueError('Warning, incompatible options selected in text parsing')

        if last_line:
            for keyword_number in list(results.keys()):
                results[keyword_number] = results[keyword_number][-1]
        if first_line:
            for keyword_number in list(results.keys()):
                results[keyword_number] = results[keyword_number][0]
        if min_value:
            for keyword_number in list(results.keys()):
                results[keyword_number] = min(results[keyword_number])

        results_to_return = []
        for key in range(len(keywords)):
            if key in list(results.keys()):
                results_to_return.append(results[key])
            else:
                if last_line or min_value or first_line:
                    results_to_return.append(None)
                else:
                    results_to_return.append([None])

        return results_to_return
