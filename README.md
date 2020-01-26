# MS-CNN-version1
## 1 Overviews
The multi-task CNN with soft labels (MS-CNN) is a novel classification model. I realize the MS-CNN based on  Alex, ResNet, and VGG, respectively. The corresponding folders are ‘based_on_Alex’, ‘based_on_ResNet’, and ‘based_on_VGG’. Here we provide two public remote sensing datasets, the AID (the paper proposing the AID is *AID: A benchmark data set for performance evaluation of aerial scene classification,https://doi.org/10.1109/TGRS.2017.2685945*), and the NR(the paper proposing the NR is *Remote Sensing Image Scene Classification: Benchmark and State of the Art, https://doi.org/10.1109/JPROC.2017.2675998*).<br>

## 2 File structure
The file structure is as follows.<br>
>MS-CNN:<br>
>>data:<br>  
>>>train<br>
>>>val<br>
>>>test<br>
>>>datagenerate.py<br>

>>public_dataset:<br>
>>>AID.zip<br>
>>>NR.rar<br>

>>based_on_Alex:<br>
>>>soft_label:<br>
>>>>BoW.py<br>
>>>>LDA.py<br>
>>>>buildsoftlabel.py<br>
>>>>dirichlet<br>

>>>Alex_train.py<br>
>>>Alex_test.py<br>
>>>Alex_feature.py<br>
>>>Alex_MS_train.py<br>
>>>Alex_MS_test.py<br>
>>>compare_result.py<br>

>>based_on_ResNet:<br>
>>>soft_label:<br>
>>>>BoW.py<br>
>>>>LDA.py<br>
>>>>buildsoftlabel.py<br>
>>>>dirichlet<br>

>>>ResNet_train.py<br>
>>>ResNet_test.py<br>
>>>ResNet_feature.py<br>
>>>ResNet_MS_train.py<br>
>>>ResNet_MS_test.py<br>
>>>compare_result.py<br>

>>based_on_VGG:<br>
>>>soft_label:<br>
>>>>BoW.py<br>
>>>>LDA.py<br>
>>>>buildsoftlabel.py<br>
>>>>dirichlet<br>

>>>VGG_train.py<br>
>>>VGG_test.py<br>
>>>VGG_feature.py<br>
>>>VGG_MS_train.py<br>
>>>VGG_MS_test.py<br>
>>>compare_result.py<br>
      
## 3 Running requirements
All codes are developed based on python 3.7, anaconda3, and the CNNs are implemented by pytorch 1.0.  Please install related libraries according the codes, which are common libraries such as numpy, PIL, matplotlib, scipy, etc.
    
## 4 Running steps
I use the MS-CNN to classify images from the NR dataset, with the ratios of the training/validation/testing are 0.1/0.1/0.8. Therefore, the parameters in the data normalization are the mean and std of the NR dataset. If you need to change the type of datasets, please re-compute the mean and std.   Furthermore, it is convenient to change the type of datasets or the ratios of training/validation/testing in the file ‘datagenerate.py’. <br>
One could realize the MS-CNN on the basis of Alex as follows.<br>
1. open file ‘data/datagenerate.py’, add the path of the target dataset at line 13, and run the file.
2. open the directory ‘based_on_Alex’.
3. run file ‘Alex_train.py’, and the training results are saved in the folder ‘train_hard’.
4. run file ‘Alex_test.py’, and the testing results are saved as ‘Alex_test.pickle’.
5. run file ‘Alex_feature.py’. The file would extract CNN features from the training and validation sets respectively. The features are saved as ‘train_feature.pickle’ and ‘val_feature.pickle’.
6. open the directory ‘based_on_Alex/soft_label’.
7. open file ‘BoW.py’ and add the path of the training set at line 26. Run the file.
9. run file ‘buildsoftlabel.py’ to generate soft labels ‘soft_label.npy’.
10. return to directory ‘based_on_Alex’.
11. run file ‘Alex_MS_train.py’, and the training results are saved in the folder ‘train_MS’.
12. run file ‘Alex_MS_test.py’, and the testing results are saved as ‘Alex_MS_test.pickle’.
13. run file ‘compare_result.py’ to compare the differences between the methods of Alex and MS-Alex.
14. The MS-CNN could also be implemented based on ResNet or VGG by using the codes in directory ‘based_on_ResNet’ or ‘based_on_VGG’.


## 5 Others
Welcome to discuss with me. My e-mail is  luojianqiao@my.swjtu.edu.cn, or 18215675028@163.com. Without my permission, using this repository for academic publication or business is not allowed . 
