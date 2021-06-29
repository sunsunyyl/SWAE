
# number_components_input: number of pseudo-inputs K
# z_size: hidden dimension

#MNIST dataset
python experiment.py  --model_name conv_wae --epochs 200 --number_components_input 4000 --z_size 40  --beta 0.5 --dataset_name dynamic_mnist  --Train --use_training_data_init 
python experiment.py  --model_name conv_wae --epochs 200 --number_components_input 4000 --z_size 40  --beta 0.5 --dataset_name dynamic_mnist  --Test  


#Fashion dataset
python experiment.py  --model_name conv_wae --epochs 200 --number_components_input 4000 --z_size 40  --beta 0.5 --dataset_name dynamic_fashion_mnist  --Train --use_training_data_init 
python experiment.py  --model_name conv_wae --epochs 200 --number_components_input 4000 --z_size 40  --beta 0.5 --dataset_name dynamic_fashion_mnist  --Test  


#Coil20 dataset
python experiment.py  --model_name conv_wae --epochs 200 --number_components_input 4000 --z_size 40  --beta 0.5 --dataset_name coil20  --Train --use_training_data_init 
python experiment.py  --model_name conv_wae --epochs 200 --number_components_input 4000 --z_size 40  --beta 0.5 --dataset_name coil20  --Test  


#CIFAR10-sub dataset
python experiment.py  --model_name conv_wae --epochs 200 --number_components_input 4000 --z_size 512  --beta 0.5 --dataset_name cifar10sub  --Train --use_training_data_init 
python experiment.py  --model_name conv_wae --epochs 200 --number_components_input 4000 --z_size 512  --beta 0.5 --dataset_name cifar10sub  --Test  
