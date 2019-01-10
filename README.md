# Course Project：EIT5111 - Advanced digital signal processing

* **Title:** Marine mammal voice recognition - right-whale-redux
* **Author:** Yiquan Lin (linyq78@mail2.sysu.edu.cn)
* **Submission date:** 7th Jan, 2019
* **Full text:** [Digital version](paper.pdf)


## Download Dataset
* Dataset: 

 [The ICML 2013 Whale Challenge - Right Whale Redux](https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux)

* Dataset structure after unzip:
```bash
 + all
   + test2
   + train2
   + sampleSubmission.csv
```


## Requirements
* Python environment (```Python 3.6.7```):
```
tensorflow==1.12.0
scikit-learn==0.20.2
scikit-image==0.14.1
opencv-python==3.4.5.20
scipy==1.2.0
```

* For installing above modules, bash in ```./``` and run:
```bash
pip install -r requirements.txt
```

## Testing: Pre-processing Single Audio and Visualization
* Ensure you correctlty install all modules, run ```input_processor_vis.py``` in shell:
```
python input_processor_vis.py
```

* following operation has been encapsulated in ```_read_and_preprocess``` function in ```input_processor.py```. In training phase, func will called by ```tf.data``` and ```tf.estimator``` automatically.


## Split Train/Val and Generate Annotation

* Unzip downloaded dataset to ```./data```, then run ```annota_generator.py``` in shell for splitting train/val and generating annotation:
```
python annota_generator.py
```
the train/val annotation will be saved at ```./data```.

* Finally we get follow directory structure:

    ```
    + data
      + model
      + test2
      + train2
      + train_annotation.txt
      + eval_annotation.txt
    + annota_generator.py
    + input_processor.py
    + input_processor_vis.py
    + train.py
    + requirements.txt
    ```

## Training and Evaluation
* Run ```train.py``` in shell, and model checkpoint/events/results will be saved ```./data/model``` (when training model on the splited train set and evaluate model on validation set).


## Reference  

　　[1] Peter J. Dugan, Christopher W. Clark, Yann André LeCun, Sofie M. Van Parijs- [ DCL System Using Deep Learning Approaches for Land-based or Ship-based Real-Time Recognition and Localization of Marine Mammals ](https://arxiv.org/ftp/arxiv/papers/1605/1605.00982.pdf)

　　[2] Mohammad Pourhomayoun, Peter Dugan, Marian Popescu, Christopher Clark - 
[Bioacoustic Signal Classification Based on Continuous Region Processing, Grid Masking and Artificial Neural Network](https://arxiv.org/ftp/arxiv/papers/1305/1305.3635.pdf)
