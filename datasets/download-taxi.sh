#!/bin/bash
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export DATA_FOLDER="gs://cloud-samples-data/ml-engine/chicago_taxi"

export GCS_TAXI_BIG=${DATA_FOLDER}/big/taxi_trips.csv
export GCS_TAXI_TRAIN_BIG=${DATA_FOLDER}/big/taxi_trips_train.csv
export GCS_TAXI_EVAL_BIG=${DATA_FOLDER}/big/taxi_trips_train.csv

export GCS_TAXI_SMALL=${DATA_FOLDER}/small/taxi_trips.csv
export GCS_TAXI_TRAIN_SMALL=${DATA_FOLDER}/small/taxi_trips_train.csv
export GCS_TAXI_EVAL_SMALL=${DATA_FOLDER}/small/taxi_trips_train.csv


mkdir -p $1 && cd $1

ROOT="$(printf "%s\n" "$(pwd)")"

if [[ $2 == 'big' ]]; then
  echo "Downloading the big dataset..."
  mkdir big
  gsutil cp ${GCS_TAXI_TRAIN_BIG} big/taxi_trips_train.csv
  gsutil cp ${GCS_TAXI_EVAL_BIG} bigtaxi_trips_eval.csv

  export TAXI_TRAIN_BIG=${ROOT}/taxi_trips_train.csv
  export TAXI_EVAL_BIG=${ROOT}/taxi_trips_eval.csv
elif [[ $2 == 'small' ]]; then
  echo "Downloading the small dataset..."
  mkdir small
  gsutil cp ${GCS_TAXI_TRAIN_SMALL} small/taxi_trips_train.csv
  gsutil cp ${GCS_TAXI_EVAL_SMALL} small/taxi_trips_eval.csv

  export TAXI_TRAIN_SMALL=${ROOT}/small/taxi_trips_train.csv
  export TAXI_EVAL_SMALL=${ROOT}/small/taxi_trips_eval.csv
else
  echo "Downloading the big dataset..."
  mkdir big
  gsutil cp ${GCS_TAXI_TRAIN_BIG} big/taxi_trips_train.csv
  gsutil cp ${GCS_TAX_EVAL_BIG} bigtaxi_trips_eval.csv

  export TAXI_TRAIN_BIG=${ROOT}/taxi_trips_train.csv
  export TAXI_EVAL_BIG=${ROOT}/taxi_trips_eval.csv

  echo "Downloading the small dataset..."
  mkdir small
  gsutil cp ${GCS_TAXI_TRAIN_SMALL} small/taxi_trips_train.csv
  gsutil cp ${GCS_TAXI_EVAL_SMALL} small/taxi_trips_eval.csv

  export TAXI_TRAIN_SMALL=${ROOT}/small/taxi_trips_train.csv
  export TAXI_EVAL_SMALL=${ROOT}/small/taxi_trips_eval.csv
fi

echo "Downloading the prediction dataset..."
mkdir prediction
gsutil cp ${DATA_FOLDER}/prediction/taxi_trips_prediction_list.json ./prediction/taxi_trips_prediction_list.json
gsutil cp ${DATA_FOLDER}/prediction/taxi_trips_prediction_dict.json ./prediction/taxi_trips_prediction_dict.json

export TAXI_PREDICTION_LIST=${ROOT}/taxi_trips_prediction_list.json
export TAXI_PREDICTION_DICT=${ROOT}/taxi_trips_prediction_dict.json

cd -


