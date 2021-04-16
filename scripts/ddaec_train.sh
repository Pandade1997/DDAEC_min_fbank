#export LD_LIBRARY_PATH='/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64:/usr/local/cudnn-8.0-linux-x64-#v6.0/lib64'
export CUDA_VISIBLE_DEVICES='0'
model_name=DDAEC_min_fbank
echo $model_name
CUDA_VISIBLE_DEVICES=3 python -u train_ddaec.py --train_list=../filelists/trainFileList_min.txt --evaluate_file=/home/panjiahui/code/Time_Domain/DDAEC/data/mixture/test/test_factory1_snr-5_seen.samp --display_eval_steps=250 --eval_plot_num=3 --model_name=DDAEC_min_fbank --width=64 --batch_size=4
#--resume_model=../models/${model_name}/${model_name}_latest.model
