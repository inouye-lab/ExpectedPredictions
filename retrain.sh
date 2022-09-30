source venv/bin/activate

data_percents="0 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"

train_model() {
  local model=$1
  local date=$2
  local source="ridge"
  python  scripts/retrain_lgc_params.py $model --prefix exp/uncertainty/$model/$source/${model}_$date/ \
    --output retrain-ard/ --data_percents $data_percents --data data/$model/$model.pklz \
    --regression --solver bayesian-ard --enforce_subsets --n-iter-pl 30
  python  scripts/retrain_lgc_params.py $model --prefix exp/uncertainty/$model/$source/${model}_$date/ \
    --output retrain-br/ --data_percents $data_percents --data data/$model/$model.pklz \
    --regression --solver bayesian-ridge --enforce_subsets --n-iter-pl 30
  python  scripts/retrain_lgc_params.py $model --prefix exp/uncertainty/$model/$source/${model}_$date/ \
    --output retrain-ridge/ --data_percents $data_percents --data data/$model/$model.pklz \
    --regression --solver auto --enforce_subsets --n-iter-pl 30
}

train_model "abalone" "20220525-011515"
train_model "delta-ailerons" "20220525-011826"
train_model "elevators" "20220525-013304"
train_model "insurance" "20220525-013220"