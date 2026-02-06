ROOT_DIR=/data/nas-gpu/wang/atong/SMART-Moonshot/nautilus

experiment="marina-jittering-test"
input_types="{hsqc,c_nmr,h_nmr,mw,mass_spec}"

for jittering in 1.5; do
  for seed in 0; do
    jittering_safe=$(echo ${jittering} | sed 's/\./-/g')
    experiment_name="${experiment}-jittering-${jittering_safe}-seed-${seed}"
    yaml="${ROOT_DIR}/jobs/jittering_test.yaml"

    # Update YAML
    sed -i "4s/.*/  name: atong-${experiment_name} /" "$yaml"
    sed -i "42s/.*/              pixi run train.marina --input_types ${input_types} --seed ${seed} --experiment_name ${experiment_name} --jittering ${jittering} /" "$yaml"

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
