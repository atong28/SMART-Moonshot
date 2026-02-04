ROOT_DIR=/data/nas-gpu/wang/atong/SMART-Moonshot/nautilus

experiment="marina-jittering-test"
input_types="{hsqc,c_nmr,h_nmr,mw,mass_spec}"
for jittering in 0.0 0.5 1.0 1.5 2.0; do
    for seed in 0 1 2; do
        jittering_safe=$(echo ${jittering} | sed 's/\./-/g')
        experiment_name="${experiment}-jittering-${jittering_safe}-seed-${seed}"
        sed -i "4s/.*/  name: atong-${experiment_name} /" ${ROOT_DIR}/jobs/jittering_test.yaml
        sed -i "42s/.*/              pixi run train.marina --input_types ${input_types} --seed ${seed} --experiment_name ${experiment_name} --jittering ${jittering} /" ${ROOT_DIR}/jobs/jittering_test.yaml
        kubectl create -f ${ROOT_DIR}/jobs/jittering_test.yaml
    done
done
