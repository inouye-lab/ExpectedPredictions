source venv/bin/activate

data_percents="0 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
log_scale="0.0000000001, 0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000"
br_cv_params="{\"lambda_init\": [$log_scale], \"alpha_init_scale\": [$log_scale]}"

train_model() {
  local model=$1
  local date=$2
  local source="ridge"
#  python  scripts/retrain_lgc_params.py $model --prefix exp/uncertainty/$model/$source/${model}_$date/ \
#    --output retrain-ard/ --data_percents $data_percents --data data/$model/$model.pklz \
#    --regression --solver bayesian-ard --enforce_subsets --n-iter-pl 30
  python  scripts/retrain_lgc_params.py $model --prefix exp/uncertainty/$model/$source/${model}_$date/ \
    --output retrain-br/ --data_percents $data_percents --data data/$model/$model.pklz \
    --regression --solver bayesian-ridge --enforce_subsets --n-iter-pl 30 --cv_params "$br_cv_params"
#  python  scripts/retrain_lgc_params.py $model --prefix exp/uncertainty/$model/$source/${model}_$date/ \
#    --output retrain-ridge/ --data_percents $data_percents --data data/$model/$model.pklz \
#    --regression --solver auto --enforce_subsets --n-iter-pl 30
}

train_model "delta-ailerons" "20220525-011826"
train_model "elevators" "20220525-013304"
train_model "insurance" "20220525-013220"
train_model "abalone" "20220525-011515"