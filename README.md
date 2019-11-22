Prototype code for replicating core results from CVPR Submission 9818. Does not include MNIST-1K, CAT128x128 and StyleGAN.

Main code contained in src/gan.py . Helper file src/run.py for feeding configuration for a run.

Toy data code in src/toy.py .

The CIFAR-10 dataset is expected in data/cifar-10 and not included in the repository.

Evaluation of metrics requires additional setup: training classifiers, using src/mnist.py and src/cifar.py, as well as including the pre-trained Inception network.

Bash script bat.sh can be edited by as raw text and includes additional details, explanations and example configurations.
