import os


############################## Directory Setting ##############################


DIR_IMAGE = 'image'
DIR_VIDEO = 'video'
DIR_MESH = 'mesh'

for directory in [DIR_IMAGE, DIR_VIDEO, DIR_MESH]:
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
