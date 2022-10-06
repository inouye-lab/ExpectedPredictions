source venv/bin/activate

data_percents="0 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
missing_percents="0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"

source="ridge"
output="results/grid_search/"

evaluate_model() {
  local model=$1
  local date=$2
  local retrain=$3
  local samples=$4
  local flags=$5

  python scripts/evaluate_dataset_uncertainty.py $model --prefix exp/uncertainty/abalone/ridge/${model}_$date/ \
    --retrain_dir $retrain --output $output --samples $samples --data data/$model/$model.pklz --classes 1 \
    --data_percents $data_percents --missing $missing_percents $flags
}

train_model "delta-ailerons" "20220525-011826" "retrain-br/20221005-175600/" 10 "--input_baseline --parameter_baseline"