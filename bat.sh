##MNIST AND CIFAR-10: python src/run.py
##See example configs below or src/run.py for options. Default values only for debugging.
##Examples assume 4 GPUs exposed to TensorFlow

##Example 3-layer fully connected MNIST runs (metrics disabled: re-enabling requires Inception and MNIST classifier networks)
##Outputs saved in out/<output_folder>_<run#>
#python src/run.py --gpus='0' --output_folder=mnist_fc3_cvpr_mm --g_cost_parameter=0.00 --g_renorm=none --g_cost=mm --d_cost=ns --d_sn=0 --g_sn=0 --g_net=dense --d_net=dense --dataset=mnist --d_layers=3 --g_layers=3 --m_dim=512 --batch_size=128 --epochs=1000 --eval_n=5 --eval_skip=1 --metrics=0 &
#python src/run.py --gpus='1' --output_folder=mnist_fc3_cvpr_ns --g_cost_parameter=0.00 --g_renorm=none --g_cost=ns --d_cost=ns --d_sn=0 --g_sn=0 --g_net=dense --d_net=dense --dataset=mnist --d_layers=3 --g_layers=3 --m_dim=512 --batch_size=128 --epochs=1000 --eval_n=5 --eval_skip=1 --metrics=0 &
#python src/run.py --gpus='2' --output_folder=mnist_fc3_cvpr_mmunit --g_cost_parameter=0.00 --g_renorm=unit --g_cost=mm --d_cost=ns --d_sn=0 --g_sn=0 --g_net=dense --d_net=dense --dataset=mnist --d_layers=3 --g_layers=3 --m_dim=512 --batch_size=128 --epochs=1000 --eval_n=5 --eval_skip=1 --metrics=0 &
#python src/run.py --gpus='3' --output_folder=mnist_fc3_cvpr_mmnsat --g_cost_parameter=1.00 --g_renorm=none --g_cost=js --d_cost=ns --d_sn=0 --g_sn=0 --g_net=dense --d_net=dense --dataset=mnist --d_layers=3 --g_layers=3 --m_dim=512 --batch_size=128 --epochs=1000 --eval_n=5 --eval_skip=1 --metrics=0 &
##For explicit gradient rescaled version of mmnsat:
#python src/run.py --gpus='3' --output_folder=mnist_fc3_cvpr_mmnsat --g_cost_parameter=0.00 --g_renorm=frac --g_cost=mm --d_cost=ns --d_sn=0 --g_sn=0 --g_net=dense --d_net=dense --dataset=mnist --d_layers=3 --g_layers=3 --m_dim=512 --batch_size=128 --epochs=1000 --eval_n=5 --eval_skip=1 --metrics=0 &


##CIFAR-10 dataset expected at data/cifar-10/data_batch_1 etc
##Available at https://www.cs.toronto.edu/~kriz/cifar.html

##Trained inception network for FID expected at 'data/inception-2015-12-05.tgz'
##Available at http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz


##Train classifiers - saves to correct destination when run from root folder
##Training the CIFAR-10 classifier takes some time
#python src/mnist.py
#python src/cifar.py

##Example Conv-4-sn CIFAR runs with metrics for linear combinations
#python src/run.py --gpus='0' --dataset=cifar --output_folder=cifar_conv4sn_cvpr_nsmmnsat0.00 --g_cost_parameter=0.00 --g_cost=js --d_cost=ns --d_sn=1 --g_net=snconv --d_net=snconv --d_layers=4 --g_layers=4 --m_dim=32 --epochs=200 --batch_size=64 --runs_n=20 --eval_n=20 &
#python src/run.py --gpus='1' --dataset=cifar --output_folder=cifar_conv4sn_cvpr_nsmmnsat0.33 --g_cost_parameter=0.33 --g_cost=js --d_cost=ns --d_sn=1 --g_net=snconv --d_net=snconv --d_layers=4 --g_layers=4 --m_dim=32 --epochs=200 --batch_size=64 --runs_n=20 --eval_n=20 &
#python src/run.py --gpus='2' --dataset=cifar --output_folder=cifar_conv4sn_cvpr_nsmmnsat0.67 --g_cost_parameter=0.67 --g_cost=js --d_cost=ns --d_sn=1 --g_net=snconv --d_net=snconv --d_layers=4 --g_layers=4 --m_dim=32 --epochs=200 --batch_size=64 --runs_n=20 --eval_n=20 &
#python src/run.py --gpus='3' --dataset=cifar --output_folder=cifar_conv4sn_cvpr_nsmmnsat1.00 --g_cost_parameter=1.00 --g_cost=js --d_cost=ns --d_sn=1 --g_net=snconv --d_net=snconv --d_layers=4 --g_layers=4 --m_dim=32 --epochs=200 --batch_size=64 --runs_n=20 --eval_n=20 &

##Convenient summary of (FID,ClassDistributions) from above experiment
#python src/read_js_fid.py out/cifar_conv4sn_cvpr nsmmnsat0.00 nsmmnsat0.33 nsmmnsat0.67 nsmmnsat1.00


##TOY PROBLEMS: python src/toy.py
##Parameter 1: 0: (1-a)NS + (0+a)MM, 1: (1-a)NS + (0+a)MM-nsat
##Parameter 2: Value of weighting factor a
##Parameter 3: Toy data (ring, spiral)
##Parameter 4: Output to out/<this folder>

##Examples for replicating paper results
##Mode frequencies are only printed to console
##Plots of distributions are saved to output folder: final plot for one run included in repository under out/repo_toy_<...>
python src/toy.py 0 0.0 ring toy_ring_ns
python src/toy.py 0 1.0 ring toy_ring_mm
python src/toy.py 0 0.0 spiral toy_spiral_ns
python src/toy.py 0 1.0 spiral toy_spiral_mm
