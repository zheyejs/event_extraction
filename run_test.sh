export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
nohup python -u main.py --config ./Config/config.cfg --device cuda:0 --t_data test --test > log_test 2>&1 &
tail -f log_test
 


