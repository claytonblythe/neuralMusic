import utils

# Save .mp3 files to random 6 second .wav snippets

def main():
    home_path = "/Volumes/passport/"
    # Save randomized samples for every song in directory containing mp3s
    base_path = home_path + "data/fma_small/"
    save_path = home_path + "data/samples/"
    utils.save_random_clips(base_path, save_path, snip_length=6.0)

    # Get melspectrograms from the short random clips, save to directory
    base_path = home_path + "data/samples/"
    save_path = home_path + "data/spectrogram_tensors/"
    utils.save_spectrogram_tensors(base_path, save_path)

    #base_path = "/Users/claytonblythe/github/neuralMusic/data/samples/"
    #save_path = "/Users/claytonblythe/github/neuralMusic/data/spectrograms/"
    #utils.save_spectrograms(base_path, save_path)

if __name__ == "__main__":
    main()
