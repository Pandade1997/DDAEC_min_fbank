export LD_LIBRARY_PATH='/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64:/usr/local/cudnn-8.0-linux-x64-v5.1/lib64'
export CUDA_VISIBLE_DEVICES='0'
model_name=DDAEC_min_asr
type=latest
root_path=/home/panjiahui/code/Time_Domain/DDAEC_min_asr
model_file=/home/panjiahui/code/Time_Domain/DDAEC_min_asr/models/DDAEC_min_asr/DDAEC_min_asr_latest.model
echo $model_name
echo $model_file
#python -u test_ddaec.py --test_list=/home/panjiahui/code/Time_Domain/DDAEC_min_asr/filelists/test_list.txt --model_file=$model_file --model_name=$model_name
#python -u assess.py --assess_list=/home/panjiahui/code/Time_Domain/DDAEC_min_asr/filelists/assess_list.txt --model_name=$model_name
#python -u assess_pesq.py --assess_list=/home/panjiahui/code/Time_Domain/DDAEC_min_asr/filelists/assess_list.txt --model_name=$model_name

python -u test_ddaec.py --test_list=/home/panjiahui/code/Time_Domain/DDAEC_min_asr/filelists/test_list.txt --model_file=/home/panjiahui/code/Time_Domain/DDAEC_min_asr/models/DDAEC_min_asr/DDAEC_min_asr_latest.model --model_name=DDAEC_min_asr
