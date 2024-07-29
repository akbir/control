#!/bin/bash

RED_BONS=(4 8 16 32)
BLUE_BONS=(4 8 16 32 64)
NUM_PROBLEMS=400

for red_bon in "${RED_BONS[@]}"; do
    for blue_bon in "${BLUE_BONS[@]}"; do
        python -m src.experiments.cost_tradeoff --bon_red="$red_bon" --bon_blue="$blue_bon" --use_cache=True --num_problems="$NUM_PROBLEMS"
        echo "Finished run with BON_RED=$red_bon and BON_BLUE=$blue_bon"
        echo "----------------------------------------"
    done
done
