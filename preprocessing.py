import utils

# Save .mp3 files to random 6 second .wav snippets

def main():
    base_path = "/Users/claytonblythe/github/neuralMusic/data/music/"
    save_path = "/Users/claytonblythe/github/neuralMusic/data/saved_samples/"
    utils.save_random_clips(base_path, save_path, snip_length=6.0)

if __name__ == "__main__":
    main()
