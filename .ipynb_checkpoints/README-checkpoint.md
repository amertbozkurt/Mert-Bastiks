# Mert-Bastiks
Working Repo for the programming course project Jan 2022.

Idea: 

Build an audio pipeline that analyses input audio date in real time and generates either
a) appealing visuals based on signal analysis that can supllement the sound experience
b) generate a set of parameters used to control a StyleGAN (https://en.wikipedia.org/wiki/StyleGAN) type neural network to generate visuals from a basis dataset to supplement the audio.


Components:
There are three distinct components to the tool that can be treated more or less separately

1) The audio part: pyAudio is a python package that provides tools to capture and analyse input audio to generate processable data from. The basis of the pipeline is a module that handles audio channels 
2) Signal analysis: A controllable toolbox (most likely via an initialization file) that derives data such as tempo, prominent frequencies, spectral analysis, i.e. performs a full signal analysis for each buffer generated by the audio engine
3) The visual engine: depending on the choice a) or b) is either an auto updated animation that updates depending on the audio input (a) or an implementation of StyleGAN that is updated for each buffer-step