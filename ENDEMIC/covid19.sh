#!/usr/bin/env bash
NAME="early_detect_actual_unskewed"
OUT="temp/$NAME"
datadir="."
# addndata="./data/covid19"
addndata="./data/graph_new"
# mkdir -p ${OUT}

# # Preprocess
# # For Semi-supervised experiments, replacing dev with test doesn't result in much difference
python3 preprocess.py --corpus graph_new --output ${OUT}/data --vocab_size 60000 --save_data "demo" --behaviour_test true
# # python preprocess.py --corpus rcv1 --output ${OUT}/data --vocab_size 50000 --save_data "demo_fastText"

echo "preprocessing done"

# # Create pre-trained word embeddings
# python3 w2v.py --input ${OUT}/data --save_data "demo" --embeddings "${datadir}/crawl-300d-2M.vec" --addndata ${addndata}
# # python w2v.py --input ${OUT}/data --save_data "" 

echo "Training"
# Train the model
python3 main.py --corpus graph_new --model LSTMEncoder --debug \
--multi_gpu --input ${OUT}/data --output_path ${OUT}/model --exp_name "early_detect_actual_behaviour" \
--use_pretrained_embeddings --nepoch_no_imprv 20 --timedistributed --behaviour_test true \
--d_units 300 --d_proj 256 --d_hidden 512 --nepochs 30 --addn_dim 14 \
--optim adam --beta1 0.0 --beta2 0.98 \
--inc_unlabeled_loss --wbatchsize 1024 --wbatchsize_unlabel 1024 --lambda_at 1.0 --lambda_vat 1.0 --lambda_entropy 1.0


# PYTHONIOENCODING=utf-8 python main.py --corpus covid19 --model LSTMEncoder --debug \
# --multi_gpu --input ${OUT}/data --output_path ${OUT}/model --exp_name "covid19_fn" \
# --use_pretrained_embeddings --nepoch_no_imprv 20 --timedistributed --d_hidden 512 --nepochs 50 \
# --optim adam --beta1 0.0 --beta2 0.98 \
# --inc_unlabeled_loss --wbatchsize 3000 --wbatchsize_unlabel 3000 --lambda_at 1.0 --lambda_vat 1.0 --lambda_entropy 1.0