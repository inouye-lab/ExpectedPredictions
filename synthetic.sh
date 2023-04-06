source venv/bin/activate

generate_and_train() {
  local model=$1
  local date=$2
  local genParams=""
  if [ "$3" = true ] ; then
    genParams="--gaussian_data data/$model/$model.pklz"
  fi

  local train_count
  local now
  printf -v now '%(%Y%m%d-%H%M%S)T' -1

  # Step 1: Generate dataset
  local sdataFolder=sdata/synthetic/
  if python scripts/generate_synthetic_dataset.py $model --folder "exp/uncertainty/$model/ridge/${model}_$date/" --classes 0 \
      --output $sdataFolder --output_id $now --train_count 1000 --valid_count 500 --test_count 500 \
      --fmap_path data/$model/fmap-$model.pickle $genParams
  then
    # Step 2: train circuit pair
    local data=$sdataFolder$model/$now/$model.pklz
    local expFolder=exp/synthetic/$model/
    if python scripts/learn_circuit_pair.py $data -o $expFolder \
        --exp-id $now --vtree balanced --psdd-n-iter 100 --n-iter-sl 100 --regression --solver auto
    then
      # Step 3: retrain with data percentages/bayesian
      local log_scale="0.0000000001, 0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000"
      if python scripts/retrain_lgc_params.py $model --prefix $expFolder$now/ --data $data \
          --output retrain/ --exp-id $now --data_percents 0.01 0.25 0.5 0.75 1.0 --regression \
          --solver bayesian-ridge --enforce_subsets --n-iter-pl 30 --cv_params "{\"lambda_init\": [$log_scale]}"
      then
        # Step 4: evaluate model
        local args="$model --prefix $expFolder$now/ --retrain_dir retrain/$now/ --data $data --classes 1 --data_percents 0.5 1.0 --missing 0 0.25 0.5 0.75 0.99 "\
"--skip_delta --include_residual_input --mse_residual --input_baseline --include_trivial --input_samples 100 --psdd_samples 100 --full_training_gaussian"
        python scripts/evaluate_dataset_uncertainty.py $args --output results/synthetic/noresid/
        python scripts/evaluate_dataset_uncertainty.py $args --output results/synthetic/resid/ --residual_missingness
      else
        echo Failed to retrain circuit parameters, giving up
      fi
    else
      echo Failed to train circuit, giving up
    fi
  else
    echo Failed to generate dataset, giving up
  fi
}

generate_and_train "delta-ailerons" "20220525-011826" true
generate_and_train "insurance" "20220525-013220" true
generate_and_train "abalone" "20220525-011515" true
generate_and_train "elevators" "20220525-013304" true