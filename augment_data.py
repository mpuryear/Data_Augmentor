import numpy as np

def augment_data(x, y, num_images, tag, img_format, output_directory):
    '''
    Generate num_images images from x with the 
    label y and store the images in directory.
    
    Images will be saved to:
    /output_directory/tag_ + index in x + _unique# + .img_format



    Inputs 
    x: x is a numpy array of images with shape (none, width, height, num_channels)
    y: y is a numpy array of labels with shape (none, 1)

    num_images: number of images to generate

    tag: string to label augmented images. Augmented images will have the name "tag_index in x_unique#.img_format" and will be saved in output_directory

    img_format: string representation of the output format. 'jpeg', 'jpg', 'png'

    output_directory: relative path to output images to
    

    Output
    return x_prime, y_prime 

    x_prime: a numpy array of the augmented images with shape (none, width, height, num_channels)
    y_prime: a numpy array of labels with shape (none, 1)
    '''

    from keras.preprocessing.image import ImageDataGenerator


    batch_size = 1 

    assert(num_images > 0)



    
    ''' 
    For further documentation of the ImageDataGenerator class see
    https://keras.io/preprocessing/image/
    '''
    datagen = ImageDataGenerator(
        rotation_range=40, # degrees we can rotate max 180
        width_shift_range=0.1, # percent width to shift
        height_shift_range=0.1, # percent height to shift
        shear_range=0.2, # angle in rotation ccw in radians
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='reflect')
    
    
    y_prime = y.copy() 
    num_added = 0
    for x_batch, y_batch in datagen.flow(x, y, batch_size=batch_size,
                                         save_to_dir=output_directory,
                                         save_prefix=tag,
                                         save_format=img_format):
        if num_added == 0:
            x_prime = x_batch
        else:
            x_prime = np.concatenate((x_prime, x_batch))

        y_prime = np.concatenate((y_prime, y_batch))

        num_added += x_batch.shape[0]

        if num_added >= num_images:
            break

    return x_prime, y_prime
            


def load_images(folder, img_size):
    '''                                                                     
    return an array containing all the images from the given folder.        
    all images are converted to RGB in channel_last format, and resized     
    to img_size x img_size                                                  
    '''

    import os
    from PIL import Image
    
    images = []
    for filename in sorted(os.listdir(folder)):
        # skip the mac generated files 
        if filename != '.AppleDouble' and filename != '.DS_Store':
            img = Image.open(os.path.join(folder, filename))
            if img is not None:
                rbgimg = Image.new("RGB", img.size)
                rbgimg.paste(img)
                rbgimg = rbgimg.resize((img_size, img_size), Image.ANTIALIAS)
                np_img = np.array(rbgimg)
                images.append(np_img)
            
    return images


def test_function():
    import os
    
    num_images_to_gen = 10
    img_format = 'jpg'
    tag = 'data_aug'
    output_dir = './test_data/data_aug/'

    if not os.path.exists(output_dir):
         os.makedirs(output_dir)
    
    img_size = 224 
    x_one = np.array(load_images('./test_data/images/with/', img_size))
    x_two = np.array(load_images('./test_data/images/without/', img_size))
    
    y_one = np.ones((x_one.shape[0], 1))
    y_two = np.zeros((x_two.shape[0], 1))

    out_dir_with = output_dir + 'with'
    if not os.path.exists(out_dir_with):
        os.makedirs(out_dir_with)
    for files in os.listdir(out_dir_with):
        if files != '.AppleDouble' and files != '.DS_Store':
            os.remove(out_dir_with + '/' + files)
        
    x_one_p, y_one_p = augment_data(x_one,
                                    y_one,
                                    num_images_to_gen,
                                    tag,
                                    img_format,
                                    out_dir_with)

    out_dir_without = output_dir + 'without'
    if not os.path.exists(out_dir_without):
        os.makedirs(out_dir_without)
        
    for files in os.listdir(out_dir_without):
        if files != '.AppleDouble' and files != '.DS_Store':
            os.remove(out_dir_without + '/' +files)
        
    x_two_p, y_two_p = augment_data(x_two,
                                    y_two,
                                    num_images_to_gen,
                                    tag,
                                    img_format,
                                    out_dir_without)
                                    
    


#test_function()
