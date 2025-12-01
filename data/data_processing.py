# -*- coding: utf-8 -*-
"""
Image pre-processing
"""

from skimage.transform import resize
import numpy as np
import h5py
import SimpleITK as sitk
import os


def min_max_norm(img):
  ''' Min max normalization of a 3D volume'''
  max_ = img.max()
  min_ = img.min()
  img = (img-min_)/(max_-min_)
  img = img.astype('float32')
  return img

def crop_bbox(inputImg, bbox_shape=(128,128,16)):
    ''' 
    Cropping a 3D bounding box around the central region of prostate MRI.
    All images are first resized to 256x256x24 - then the center 16 slices containing 
    the prostate are retained.
    '''
    cropped_img = resize(inputImg, (256,256,bbox_shape[2]+8))
    cropped_img = myCrop3D(cropped_img,(bbox_shape[0],bbox_shape[1]))
    cropped_img = cropped_img[...,4:-4]
    return cropped_img


def myCrop3D(ipImg,opShape):
    '''
    Custom 3D in-plane crop function. A given input image is cropped to the provided output size (xDim, yDim).
    Zero padding is performed when the output size is larger than the input size.
    Cropping is performed with the output size is smaller than the input size. 
    The z-dimension is retained as-is
    '''
    xDim,yDim = opShape
    zDim = ipImg.shape[2]
    opImg = np.zeros((xDim,yDim,zDim))
    
    xPad = xDim - ipImg.shape[0]
    yPad = yDim - ipImg.shape[1]
    
    x_lwr = int(np.floor(np.abs(xPad)/2))
    x_upr = int(np.ceil(np.abs(xPad)/2))
    y_lwr = int(np.floor(np.abs(yPad)/2))
    y_upr = int(np.ceil(np.abs(yPad)/2))
    if xPad >= 0 and yPad >= 0:
        opImg[x_lwr:xDim - x_upr ,y_lwr:yDim - y_upr,:] = ipImg
    elif xPad < 0 and yPad < 0:
        xPad = np.abs(xPad)
        yPad = np.abs(yPad)
        opImg = ipImg[x_lwr: -x_upr ,y_lwr:- y_upr,:]
    elif xPad < 0 and yPad >= 0:
        xPad = np.abs(xPad)
        temp_opImg = ipImg[x_lwr: -x_upr,:,:]
        opImg[:,y_lwr:yDim - y_upr,:] = temp_opImg
    else:
        yPad = np.abs(yPad)
        temp_opImg = ipImg[:,y_lwr: -y_upr,:]
        opImg[x_lwr:xDim - x_upr,:,:] = temp_opImg
    return opImg

def get_images_from_h5py(filename):
    ''' 
    Retrieve images from a h5 filename. 
    The h5 structure is as described in ./data
    '''
    # Get t2w and diffusion volumes for a subject from h5 files    
    with h5py.File(filename, 'r') as f:
        t2 = f['axt2'][:]
        adc = f['adc'][:]
        b1500 = f['b1500'][:]
    
    # Process t2w data
    t2_crop = min_max_norm(t2)
    t2_crop = crop_bbox(t2)
    t2 = t2_crop.squeeze()
    axt2_img = np.expand_dims(t2, axis=-1)

    # Process diffusion data
    adc = min_max_norm(adc)
    b1500 = min_max_norm(b1500)
    adc_crop = crop_bbox(adc)
    b1500_crop = crop_bbox(b1500)
    diff_img = np.stack((adc_crop, b1500_crop), axis=-1)
    return axt2_img, diff_img


def resample_img(image, out_spacing):
    """
    Resample images to target resolution spacing
    Ref: SimpleITK
    """
    # get original spacing and size
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()      
    out_spacing = list(out_spacing) 
    # calculate output size in voxels
    out_size = [
        int(np.round(
            size * (spacing_in / spacing_out)
        ))
        for size, spacing_in, spacing_out in zip(original_size, original_spacing, out_spacing)
    ]
 
    # set up resampler
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(list(out_spacing))
    resample.SetSize(out_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    # perform resampling
    image = resample.Execute(image)
    return image

def get_images_from_mha(filename, data_type, bbox_shape=(128,128,16)):
    ''' 
    Retrieve images from .mha files (e.g., PI-CAI dataset)
    '''

    if data_type == 'axt2':
        target_resolution = (0.6, 0.6, 3.0)
    else:
        target_resolution = (2.0, 2.0, 3.0)
    image = sitk.ReadImage(filename)
    image = resample_img(image,(target_resolution))
    image = sitk.GetArrayFromImage(image)
    image = min_max_norm(np.transpose(image,(1,2,0)))
    image = crop_bbox(image, bbox_shape)
    return image

def get_images_demo(picai_dir, patient_num,study_num, bbox_shape=(128,128,16)):
    '''
    Return axt2 and diff MR data from PI-CAI.
    Expects directory, patient num and study number
    '''
    filename = os.path.join(picai_dir,str(patient_num), str(patient_num) + '_' + str(study_num) + '_t2w.mha')
    image = sitk.ReadImage(filename)
    axt2 = resample_img(image,(0.6,0.6,3))
    filename = os.path.join(picai_dir,str(patient_num), str(patient_num) + '_' + str(study_num)  + '_hbv.mha')
    b1500 = sitk.ReadImage(filename)
    b1500 = resample_img(b1500,(2.0,2.0,3))
    filename = os.path.join(picai_dir,str(patient_num), str(patient_num) + '_' + str(study_num)  + '_adc.mha')
    adc = sitk.ReadImage(filename)
    adc = resample_img(adc,(2.0,2.0,3))
    axt2 = sitk.GetArrayFromImage(axt2)
    b1500 = sitk.GetArrayFromImage(b1500)
    adc = sitk.GetArrayFromImage(adc)

    axt2 = min_max_norm(np.transpose(axt2,(1,2,0)))
    b1500 = min_max_norm(np.transpose(b1500,(1,2,0)))
    adc = min_max_norm(np.transpose(adc,(1,2,0)))
    axt2_img = crop_bbox(axt2,bbox_shape)
    b1500_crop = crop_bbox(b1500,bbox_shape)
    adc_crop = crop_bbox(adc,bbox_shape)
    diff = np.stack((adc_crop,b1500_crop), axis=-1)
    return axt2_img, diff


def get_imaging_priors(current_accnum, df, data_dir, num_timepoints_to_use=3):
    '''
    Given curr accession number, return imaging priors from upto as many timepoints that exist.
    Assumes all data is available in data_dir, processed either as .h5, .niis , or .mha

    Returns Prior MRIs, prior time intervals, current MRI, current time
    '''

    curr_filename = os.path.join(data_dir, str(current_accnum) + '.h5' ) 
    print(f'Loading current imaging data from {current_accnum}')

    curr_axt2, curr_diff = get_images_from_h5py(curr_filename)

    # Identify previous accession number from the pandas df

    temp_df = df.loc[df['Curr_AccNum']==current_accnum]
    priors_available = temp_df['Previous_AccNum'].values[0]
    num_priors_available = len(priors_available)
    curr_age = temp_df['Curr_Age'].values[0]
    
    curr_img = [curr_axt2[np.newaxis,...], curr_diff[np.newaxis,...]] 
    # current time set to 1, prior times set to age difference between current and prior ages
    ti_recent = np.ones((1,)).astype('int32')
 
    if num_priors_available < num_timepoints_to_use:
        select_idx = np.arange(num_priors_available)  # indices to select
        ti_diff = np.zeros((num_priors_available,))   # time difference initialization (yrs)
    else:
        select_idx = np.arange(num_timepoints_to_use)
        ti_diff = np.zeros((num_timepoints_to_use,))
   
    prev_acc_num_seq = temp_df['AccessionNumberList'].values[0]
    prev_age_seq = temp_df['Age'].values[0]
    prev_axt2 = []
    prev_diff = []
    
    count = 0
    for idx in select_idx:
        prev_accnum = prev_acc_num_seq[idx]
        print(f'Loading prior imaging data from {prev_accnum}')
        age = prev_age_seq[idx]
        prev_filename = os.path.join(data_dir, str(prev_accnum) + '.h5' ) 
        axt2, diff  = get_images_from_h5py(prev_filename)
        prev_axt2.append(axt2[np.newaxis,...])
        prev_diff.append(diff[np.newaxis,...])
        ti_diff[count] = curr_age - age
        count = count+1
    
    ti_prior = ti_diff + ti_recent
    ti_prior = ti_prior.astype('int32')
    # Time difference is clipped to 10 years
    ti_prior = np.clip(ti_prior,0, 9)
    prev_axt2 = np.concatenate(prev_axt2, axis=0)
    prev_diff = np.concatenate(prev_diff, axis=0)
    prior_img = [prev_axt2, prev_diff] 

    return prior_img, ti_prior, curr_img, ti_recent 

def get_imaging_priors_demo(current_accnum, df, data_dir, num_timepoints_to_use=3):
    '''
    Given curr accession number, return imaging priors from upto as many timepoints that exist.
    Assumes all data is available in data_dir, processed either as .h5, .niis , or .mha

    Returns Prior MRIs, prior time intervals, current MRI, current time
    '''

    temp_df = df.loc[df['Curr_AccNum']==current_accnum]
    patient_num = temp_df['PatientID'].values[0]
    priors_available = temp_df['Previous_AccNum'].values[0]
    num_priors_available = len(priors_available)
    curr_age = temp_df['Curr_Age'].values[0]

 
    print(f'Loading current imaging data from {current_accnum}')

    curr_axt2, curr_diff = get_images_demo(data_dir, patient_num,current_accnum, bbox_shape=(128,128,16))
    curr_img = [curr_axt2[np.newaxis,...], curr_diff[np.newaxis,...]] 
    
    # current time set to 1, prior times set to age difference between current and prior ages
    ti_recent = np.ones((1,)).astype('int32')
 
    if num_priors_available < num_timepoints_to_use:
        select_idx = np.arange(num_priors_available)  # indices to select
        ti_diff = np.zeros((num_priors_available,))   # time difference initialization (yrs)
    else:
        select_idx = np.arange(num_timepoints_to_use)
        ti_diff = np.zeros((num_timepoints_to_use,))
   
    prev_acc_num_seq = temp_df['Previous_AccNum'].values[0]
    prev_age_seq = temp_df['Previous_Age'].values[0]
    prev_axt2 = []
    prev_diff = []
    
    count = 0
    for idx in select_idx:
        prev_accnum = prev_acc_num_seq[idx]
        print(f'Loading prior imaging data from {prev_accnum}')
        age = prev_age_seq[idx]
        axt2, diff  = get_images_demo(data_dir, patient_num, prev_accnum, bbox_shape=(128,128,16))
        prev_axt2.append(axt2[np.newaxis,...])
        prev_diff.append(diff[np.newaxis,...])
        ti_diff[count] = curr_age - age
        count = count+1
    
    ti_prior = ti_diff + ti_recent
    ti_prior = ti_prior.astype('int32')
    # Time difference is clipped to 10 years
    ti_prior = np.clip(ti_prior,0, 9)
    prev_axt2 = np.concatenate(prev_axt2, axis=0)
    prev_diff = np.concatenate(prev_diff, axis=0)
    prior_img = [prev_axt2, prev_diff] 

    return prior_img, ti_prior, curr_img, ti_recent 