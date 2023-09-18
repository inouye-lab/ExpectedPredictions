source venv/bin/activate

train() {
  local model=$1
  local date=$2
  local retrain=$3
  python scripts/learn_neural_network.py data/$model/$model.pklz --n-iter 200 -o ./nn/$model/ \
    --data_percents 0 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --enforce_subsets \
    --sample_index_folder exp/uncertainty/$model/ridge/${model}_$date/$retrain/
}

train "abalone" "20220525-011515" "retrain-br/20221005-235937"
train "delta-ailerons" "20220525-011826" "retrain-br/20221006-125008"
train "elevators" "20220525-013304" "retrain-br/20221005-202956"
train "insurance" "20220525-013220" "retrain-br/20221006-131629"