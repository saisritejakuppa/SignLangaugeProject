import pandas as pd
import requests
import os
import subprocess
import pickle
import cv2
import shutil
import gzip
import json
from tqdm import tqdm

def GetDataFromCSV(csv_file):
    """Read data from csv file.

    Args:
        csv_file: csv file path.

    Returns:
        data: data read from csv file.
    """
    data = pd.read_csv(csv_file, sep='|')
    return data



def DownloadVideo(url, savepath):
    response = requests.get(url)

    if response.status_code == 200:
        with open(savepath, 'wb') as f:
            f.write(response.content)
            print("Video saved successfully!")
        return True
    else:
        print("Failed to download video.")
        return False


def DownloadOpenPose(url, savepath):
    response = requests.get(url)

    if response.status_code == 200:
        with open(savepath, 'wb') as f:
            f.write(response.content)
            print("OpenPose gz saved successfully!")
        return True
    else:
        print("Failed to download OpenPose.")
        return False




def makedir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    else:
        pass




def DownloadDataset(dataframe, donwloaddir = '../outputs/dataset_preparation/raw_vids_dataset'):


    #make the directory if doesnot exist
    makedir(donwloaddir)

    unq_filenames = sorted(list(set(dataframe['filename'])))

    #split the filename by -
    unq_filenames = [filename.split('-')[0][:-1] for filename in unq_filenames]

    for unq_filename in unq_filenames:
        vid_base_url = f'https://www.sign-lang.uni-hamburg.de/meinedgs/videos/{unq_filename}/{unq_filename}.mp4'
        pose_base_url = f"https://www.sign-lang.uni-hamburg.de/meinedgs/openpose/{unq_filename}_openpose.json.gz"

        vid_savepath = os.path.join(donwloaddir, unq_filename,f'{unq_filename}.mp4')
        pose_savepath = os.path.join(donwloaddir, unq_filename,  f'{unq_filename}_openpose.json.gz')

        makedir(os.path.join(donwloaddir, unq_filename))

        #download the url
        DownloadVideo(vid_base_url, vid_savepath)
        DownloadOpenPose(pose_base_url, pose_savepath)

        savepath = pose_savepath.replace('openpose.json.gz', 'openpose.json')
        UnzipGZFile(pose_savepath, savepath)

        #remove the pose_savepath
        os.remove(pose_savepath)

        # break

    # print(unq_filenames[0])

def GetFps(vidpath):
    cap = cv2.VideoCapture(vidpath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps



def GetTimeFormat(starttime, fps):

    # Calculate the number of seconds
    seconds = starttime / fps

    # Calculate the number of minutes
    minutes, seconds = divmod(seconds, 60)

    # Calculate the number of hours
    hours, minutes = divmod(minutes, 60)

    # Format the modified start time
    modified_start = '{:02d}:{:02d}:{:02d}:{}'.format(int(hours), int(minutes), int(seconds), int(starttime % fps))

    return modified_start


def UnzipGZFile(gzfilepath, savepath):
    with gzip.open(gzfilepath, 'rb') as f_in:
        with open(savepath, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)



def extract_frames_info(json_file, start_frame, end_frame, frame):
    # Load the JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)

    info = []

    # Loop through the data
    for item in data:
        if item['camera'] == frame:
            # Extract the information between frames
            frames = item['frames']
            for frame in frames:
                if start_frame <= int(frame) <= end_frame:
                    # print(f"Frame: {frame}")
                    # print(f"Information: {frames[frame]}")

                    info.append(frames[frame])
    
    return info

# Call the function
def GenerateDatasetFromVid(specific_vid_dataframe, donwloaddir):

    for index, row in tqdm(specific_vid_dataframe.iterrows()):

        unq_filename = row['filename']
        
        filename = row['filename'].split('-')[0][:-1]
        start_frame = row['start_time']
        end_frame = row['stop_time']

        vidname = os.path.join(donwloaddir, filename, f'{filename}.mp4')
        jsonname = os.path.join(donwloaddir, filename, f'{filename}_openpose.json')

        fps = GetFps(vidname)

        modified_start = GetTimeFormat(start_frame, fps)
        modified_end = GetTimeFormat(end_frame, fps)
        print(modified_start, modified_end)

        # Convert the modified start time to seconds
        start_seconds = int(modified_start[0:2]) * 3600 + int(modified_start[3:5]) * 60 + int(modified_start[6:8]) + int(modified_start[9:11]) / 100

        # Convert the modified end time to seconds
        end_seconds = int(modified_end[0:2]) * 3600 + int(modified_end[3:5]) * 60 + int(modified_end[6:8]) + int(modified_end[9:11]) / 100

        savepath = os.path.join(donwloaddir, filename, unq_filename ,f'{unq_filename}.mp4')
        makedir(os.path.join(donwloaddir, filename, unq_filename))

        subprocess.run(['ffmpeg', '-i', vidname, '-ss', str(start_seconds), '-to', str(end_seconds), '-c:v', 'copy', '-c:a', 'copy', savepath])


        final_vid = savepath.replace('.mp4', '_final.mp4')
        resized_vid = savepath.replace('.mp4', '_resized.mp4')

        if row['camera'] == 'A':
            subprocess.run(['ffmpeg', '-i', savepath, '-filter:v', 'crop=640:ih:0:0', final_vid])
            subprocess.run(['ffmpeg', '-i', final_vid, '-filter:v', 'scale=1920:1080', resized_vid])
            pose_info = extract_frames_info(jsonname, start_frame, end_frame, 'a1')
            pass
            
        
        if row['camera'] == 'B':
            subprocess.run(['ffmpeg', '-i', savepath, '-filter:v', 'crop=640:ih:640:0', final_vid])
            subprocess.run(['ffmpeg', '-i', final_vid, '-filter:v', 'scale=1920:1080', resized_vid])
            pose_info = extract_frames_info(jsonname, start_frame, end_frame, 'b1')
            pass

        #remove the savepath
        os.remove(savepath)
        os.remove(final_vid)

        #save the pose_info to a pickle
        savepath = os.path.join(donwloaddir, filename, unq_filename ,f'{unq_filename}.pkl')
        makedir(os.path.join(donwloaddir, filename, unq_filename))
        with open(savepath, 'wb') as f:
            pickle.dump(pose_info, f)

        # break



def GenerateFrameDataset(dataframe, donwloaddir = '../outputs/dataset_preparation/raw_vids_dataset'):
    downloaded_vids = os.listdir(donwloaddir)
    for vid in tqdm(downloaded_vids):
        #get the modified dataframe by checking if the name of the vid is in dataframe
        modified_df = dataframe[dataframe['filename'].str.contains(vid.split('.')[0])]
        GenerateDatasetFromVid(modified_df, donwloaddir)
        print(modified_df.shape)
    return 





if __name__ == "__main__":

    train_path = '../meineDGS-Translation-Protocols/mDGS/mDGS_Protocol_Train.csv'
    data = GetDataFromCSV(train_path)

    DownloadDataset(data, '../outputs/dataset_preparation/raw_vids_dataset')
    GenerateFrameDataset(data, '../outputs/dataset_preparation/raw_vids_dataset')
    # print(data.head())


