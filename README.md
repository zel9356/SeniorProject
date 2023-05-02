# SeniorProject
There are several paprmenters to chnage in differnt spots
  Threshold -> Change T at top of transformGraph.py 
  
  The k values in kNNs -> change 1_1max - k_6min in the function adaptive_gw in DWkNNFromROI.py
  
  Channels used -> chnage what files are in the folders of input images
  
  Number of dimensions in the lower dimension space -> Change L at top of transformGraph.py
  
  
test_with_set_v_main.py currently runs the me6thod on sevral images using the same V. V can be changed by changing what file is read in as set_v_file. 


The call to run each image is as follows
testWithSetV(set_v, folder_of_imgs, save_name)
    :param set_v: file of set v values
    :param folder_of_imgs: folder containing multi-spectral images
    :param save_name: name for saving image
    :return: 
    
  To get a file containg the V info, run presetV, changing the file name in the main function to the folder of multispectral images to be used for V
