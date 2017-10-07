import nmutils
import argparse

# Save .mp3 files to random 6 second .wav snippets

def get_args():
    '''This function parses and return arguments passed in'''
    # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description='Script that process fma music to spectrogram tensors')
    # Add arguments
    parser.add_argument(
        '-s', '--size', type=str, help='Size directory name', default='small')
    # Array for all arguments passed to script
    args = parser.parse_args()
    # Assign args to variables
    size = args.size
    # Return all variable values
    return(size)


def main():
    home_path = "/Volumes/passport/"
    # Match return values from get_arguments()
    size = get_args()

    # Save master genre list
    base_path = home_path + "data/fma_metadata/"
    save_path = home_path + "data/"
    nmutils.save_genre_master_list(base_path, save_path)
    # Save randomized samples for every song in directory containing mp3s
    base_path = home_path + "data/fma_" + size + '/'
    save_path = home_path + "data/samples/"
    nmutils.save_random_clips(base_path, save_path, snip_length=6.0)
    # Get melspectrograms from the short random clips, save to directory
    base_path = home_path + "data/samples/"
    save_path = home_path + "data/spectrogram_tensors/"
    nmutils.save_spectrogram_tensors(base_path, save_path)
    # Save csv file of the current spectrogram_tensor genres
    base_path = home_path + "data/spectrogram_tensors/"
    save_path = home_path + "data/"
    nmutils.save_tensor_labels(base_path, save_path)

if __name__ == "__main__":
    main()
