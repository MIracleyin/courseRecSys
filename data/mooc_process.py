import bz2
import csv
import json
import operator
import os
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from data.base_dataset import BaseDataset


class MOOCProcess(BaseDataset):
    def __init__(self, input_path, output_path, dataset_name='mooc'):
        super(MOOCProcess, self).__init__(input_path, output_path)
        self.dataset_name = dataset_name
        self.inter_file = os.path.join(self.input_path, "data.csv")
        self.sep = ","
        self.inter_fields = {0: 'stu_id:token',
                             1: 'course_id:token',
                             2: 'timestamp:float'}  # useful col

        self.output_inter_file = self._get_output_files()

    def _get_output_files(self):
        output_inter_file = os.path.join(self.output_path, self.dataset_name + '.inter')
        return output_inter_file

    def load_inter_data(self):
        origin_data = pd.read_csv(self.inter_file, delimiter=self.sep, engine='python', encoding='gb18030')
        origin_data = origin_data.iloc[:, [0, 2, 1]]  # get stu_id course_id timestamp
        origin_data.iloc[:, 2] = origin_data.iloc[:, 2].apply(lambda x: pd.Timestamp(x).to_julian_date())
        return origin_data

if __name__ == '__main__':
    l = MOOCProcess('../dataset/mooc_data', '../dataset/mooc')
    l.convert_inter()
