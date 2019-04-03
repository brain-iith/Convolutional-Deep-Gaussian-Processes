# Deep Gaussian Processes with Convolutional Kernels

## Important files and scripts

* **gpflowrc** is a setting file. It helps in fixing float type(extremely useful in case of cholesky error). To fix cholesky error, add **jitter** or increase it.
* **wconv_dgp_train.py** file is the one which is needed to be run. 
* As file runs, credentials such as nlpp,accuracy and time/epochs will be saved in separate folder(at step size of 5 epochs, user can change that in optimizer function).
* **loading_model.py** is an example of loading a pretrained model.
* **‘data’** directory contains the data. It contains MNIST dataset and example files for convex non convex dataset.

## Saving Model

* Get parameters of a model in a dictionary:  

     param_dict = m_dgp2.read_trainables()
* Saving parameters as dictionary into a file using pickle :  

     import pickle  
     f=open(“file.pkl”,”wb”)  
     pickle.dump(param_dict,f)  
     f.close()
     
## Loading Model

* Loading dictionary back from pickle file: 

     param_dict = pickle.load(open(“file.pkl”,”rb”))
* Assigning parameters to the model using loaded dictionary:  

     make a similar model, say, my_model, then do  
     my_model.assign(param_dict)

## Dependencies

* Python 3
* Tensorflow v1.10
* GPflow v1.1

