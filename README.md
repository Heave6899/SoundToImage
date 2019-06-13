# SoundToImage
It is used to convert sound .wav files to .png files and analyse them
The code is fairly simple to recreate and uses the following prerequisite libraries in python:
  1. pydub - sound connversion of any file to 16 bit wav file
  2. PIL - for conversion of numpy array of H x W x 3 shape to RGB image of png extension
  3. scipy.io - to import a sound file
  4. numpy - to access and store the returned array of PIL extractor function
