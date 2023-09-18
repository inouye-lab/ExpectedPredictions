source venv/bin/activate

source="ridge"
output="results/nn-bool3"

evaluate_model() {
  local model=$1
  local date=$2
  local retrain=$3
  local nn_date=$4
  local flags=$5

  python scripts/evaluate_dataset_uncertainty.py $model --prefix exp/uncertainty/$model/ridge/${model}_$date/ \
    --retrain_dir $retrain --output $output --data data/$model/$model.pklz --classes 1 $flags \
    --fmap data/$model/fmap-$model.pickle --nn_folder nn/$model/${model}_$nn_date/
}

common="--skip_delta --include_residual_input --mse_residual --input_baseline --include_trivial --input_samples 100 --psdd_samples 100 --full_training_gaussian --include_conditional"
data_percents="0.5 1.0"
missing_percents="0 0.25 0.5 0.75 0.99"

model="abalone"
training="20220525-011515"
retraining="retrain-br/20221005-235937/"
nn_date="20230509-040228"
evaluate_model $model $training $retraining $nn_date "--data_percents $data_percents --missing $missing_percents $common"
evaluate_model $model $training $retraining $nn_date "--data_percents $data_percents --missing $missing_percents $common --residual_missingness"

model="insurance"
training="20220525-013220"
retraining="retrain-br/20221006-131629/"
nn_date="20230509-042231"
evaluate_model $model $training $retraining $nn_date "--data_percents $data_percents --missing $missing_percents $common"
evaluate_model $model $training $retraining $nn_date "--data_percents $data_percents --missing $missing_percents $common --residual_missingness"

model="delta-ailerons"
training="20220525-011826"
retraining="retrain-br/20221006-125008/"
nn_date="20230509-040804"
evaluate_model $model $training $retraining $nn_date "--data_percents $data_percents --missing $missing_percents $common"
evaluate_model $model $training $retraining $nn_date "--data_percents $data_percents --missing $missing_percents $common --residual_missingness"

model="elevators"
training="20220525-013304"
retraining="retrain-br/20221005-202956/"
nn_date="20230509-041454"
evaluate_model $model $training $retraining $nn_date "--data_percents $data_percents --missing $missing_percents $common"
evaluate_model $model $training $retraining $nn_date "--data_percents $data_percents --missing $missing_percents $common --residual_missingness"
