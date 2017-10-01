

This is a project called neuralMusic created by Clayton Blythe on 2017/09/29

I aim to use the Free Music Archive (FMA) along with Convolutional Neural Networks to do genre classification from 
short snippets of different songs (~6 seconds of audio). I am working on taking random 6 second samples from 100,000 songs to train a classifier. 

Here is an example of what a spectrogram looks like, it is kind of like a "fingerprint" for a song, representing how different frequencies of sound evolve over time. 

For example, here is a six second snippet from "Lose Yourself To Dance" by Daft Punk

![Alt Test](https://github.com/claytonblythe/neuralMusic/blob/master/data/spectrograms/lose_yourself_to_dance.png)


Convolutional Neural Networks have contributed to amazing advancements in image recognition, and this dataset is fairly large, so I am looking to see how good they are at converting the visual representation of a snippet of audio into genre predictions. I imagine it could be a cool app where you can get classification of a genre from a very short recording of audio


