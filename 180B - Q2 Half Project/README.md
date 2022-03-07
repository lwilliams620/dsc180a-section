# Requirements
`pip install tensorflow==2.6`

`pip install keras==2.6`

`pip install numpy`

`pip install packaging`

`pip install larq`


# Instructions
1. Pull and run the docker file `l6willia/dsc180b`
2. Clone this repository
3. Run `python3 <file name>.py`

# Code Explanation
## quantize.py
Contains helper code to quantize an lstm and to modify the bits used in quantization.

## run.py
Contains code for a MLP on the MNIST dataset. To change the number of bits used, change the variables at the top of the file. By default, it runs with 1 bit.

## lstm_binary.py
Contains code for a quantized LSTM on IMDb data for sentiment analysis.
