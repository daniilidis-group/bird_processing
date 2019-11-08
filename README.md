# bird processing

This repo has code for processing the bird data, in particular handling the
compressed HEVC/mp4 files.

## pytorch video loader

The ``SynchronizedVideoLoader`` class lets you recover synchronized
data from an unsynchronized video stream. But first you need to get
the video into shape:

    bag_file=full_path_to_my_bag_file.bag
	video_dir=full_path_to_my_target_video_directory
	roslaunch ffmpeg_image_transport_tools split_bag.launch	bag:=$bag_file out_file_base:=$video_dir/video_ write_time_stamps:=true convert_to_mp4:=true

Running this should produce 8 videos in the video output directory, along with time stamp files that map frame numbers to ROS time stamps.

Once you have the files in place, you can use a
``SynchronizedVideoLoader`` to access it in pytorch format:


    loader = SynchronizedVideoLoader(file_root_dir, crop_size)

    for i, inputs in enumerate(loader):
        frame_0 = inputs[0]
        print(i, 'size: ', frame_0['data'].shape, frame_0['time'])


There is a little demo program you can check out:

    ./src/test_bird_video_loading.py -r $video_dir

## projecting points

In the ``src`` directory there is the demo code ``projector.py`` that projects
3d points into cameras. It needs the ``calibration.yaml`` file that is produced
during extrinsic calibration 
