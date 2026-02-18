ROOT_DIR=/data/nas-gpu/wang/atong/SMART-Moonshot/nautilus

experiment="marina-dataset-experiments"
input_types="{hsqc,c_nmr,h_nmr,mw}"

for dataset in MARINAControl1 MARINABase1 MARINADataset1 MARINADataset2 MARINADataset3 MARINADataset4; do
  for seed in 0 1 2; do
    dataset_lowercase=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
    experiment_name="${experiment}-dataset-${dataset_lowercase}-seed-${seed}"
    yaml="${ROOT_DIR}/jobs/dataset_experiments.yaml"
    # Update YAML
    sed "4s|.*|  name: atong-${experiment_name} |" "$yaml" > "$yaml.tmp" && mv "$yaml.tmp" "$yaml"
    sed "40s|.*|              bash /root/gurusmart/startup.sh ${dataset}.zip |" "$yaml" > "$yaml.tmp" && mv "$yaml.tmp" "$yaml"
    sed "42s|.*|              pixi run train.marina --input_types ${input_types} --seed ${seed} --experiment_name ${experiment_name} |" "$yaml" > "$yaml.tmp" && mv "$yaml.tmp" "$yaml"

    job_name="atong-${experiment_name}"
    # Check if job exists
    if kubectl get job "$job_name" >/dev/null 2>&1; then
      failed=$(kubectl get job "$job_name" -o jsonpath='{.status.failed}')
      active=$(kubectl get job "$job_name" -o jsonpath='{.status.active}')

      failed=${failed:-0}
      active=${active:-0}

      if [[ "$failed" -gt 0 ]]; then
        echo "[FAILED] job/$job_name → deleting and recreating..."
        kubectl delete job "$job_name"
        kubectl create -f "$yaml"
      elif [[ "$active" -gt 0 ]]; then
        echo "[RUNNING] job/$job_name → skipping."
      else
        echo "[EXISTS] job/$job_name (not failed, not active) → skipping."
      fi
    else
      echo "[MISSING] job/$job_name → creating..."
      kubectl create -f "$yaml"
    fi

  done
done
