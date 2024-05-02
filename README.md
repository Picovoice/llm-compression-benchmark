# LLM Compression Benchmark

Made in Vancouver, Canada by [Picovoice](https://picovoice.ai)

This repo is a minimalist and extensible framework for benchmarking different LLM compression algorithms.

## Table of Contents
- [Algorithms](#algorithms)
  - [GPTQ](#gptq)
  - [picoLLM Compression](#picollm-compression)
- [Tasks](#tasks)
  - [C4 Perplexity](#c4-perplexity)
  - [ARC](#arc)
- [Data](#data)
  - [Quantization](#quantization)
  - [C4](#c4)
  - [ARC](#arc)
- [Models](#models)
- [Usage](#usage)
- [Results](#results)
  - [Perplexity](#perplexity)
  - [ARC-Easy](#arc-easy)
  - [ARC-Challenge](#arc-challenge)

## Algorithms

### GPTQ

[GPTQ](https://arxiv.org/abs/2210.17323) is arguably the most popular quantization technique for LLMs at the moment. It
is fairly powerful as it fully reconstructs the weights to closely mimic the floating-point version.  

### picoLLM Compression

picoLLM Compression is a developed by Picovoice. What sets it apart is that it optimally distributes bits (resources)
within and across model parameters. picoLLM accepts a target model size and given that distributes all available bits
optimally across and within model parameters. Hence, picoLLM is an x-bit quantization technique. 

## Tasks

### C4 Perplexity

Perplexity is very sensitive to quantization and can be used to detect deterioration early on. It is a language modeling
task.

### ARC

[AI2 Reasoning Challenge (ARC) dataset](https://allenai.org/data/arc) is a multiple choice dataset that can measure the
models ability to perform reasoning. ARC dataset is partitioned into two segments: easy and challenge. We perform the benchmark
on both partitions and report the results separately.

## Data

All the data needed to run the benchmark is already available under [res](res) for ease of use. But if you wish to reproduce
it or find out how the data is curated or even change it you can use the sections below:

### Quantization

We do need a sample dataset for GPTQ and picoLLM to learn characteristics of the model to perform their algorithms. We
choose to use 128 randomly selected sequences from the train portion of the [C4 dataset](https://huggingface.co/datasets/c4). Once you download the dataset
run the following from the root of the repository to extract and normalize the data:

```console
python3 data/c4-normalize.py --repository-folder ${REPOSITORY_FOLDER} --normalized-folder ${TRAIN_FOLDER} --portion train
```
replace `${REPOSITORY_FOLDER}` with the path the downloaded dataset repository, `${TRAIN_FOLDER}` with a folder to hold on to
the normalized data.

Then we sample 128 sequences from teh normalized data:

```console
python3 data/c4-sample.py --dataset-folder ${TRAIN_FOLDER} --portion train
```

### C4

For the perplexity task we use 128 randomly selected snippets from the validation portion of the 
[C4 dataset](https://huggingface.co/datasets/c4). Once you download the dataset  run the following from the root of the
repository to extract and normalize the data:

```console
python3 data/c4-normalize.py --repository-folder ${REPOSITORY_FOLDER} --normalized-folder ${VALIDATION_FOLDER} --portion validation
```

Then we sample 128 sequences from teh normalized data:

```console
python3 data/c4-sample.py --dataset-folder ${VALIDATION_FOLDER} --portion valid
```

### ARC

[ARC dataset](https://allenai.org/data/arc)

```console
python3 data/arc.py  --dataset-folder ${DATASET_FOLDER}
```

```console
python3 data/arc.py  --dataset-folder ${DATASET_FOLDER} --easy
```

### Models

- `Llama-3-8b`
- `Llama-2-7b`
- `Gemma-2b`
- `Gemma-7b`
- `Phi-2`
- `Mistral-7b-v0.1`

[Picovoice Console](https://console.picovoice.ai/)

```console
python3 model/autogptq.py --model-uri ${MODEL_URI} --quantized-model-folder ${QUANTIZED_MODEL_FOLDER} --bits ${BITS}
```

## Usage

```console
python3 perplexity.py --compression ${COMPRESSION} --model-uri ${MODEL_URI}
```

```console
python3 arc.py --compression ${COMPRESSION} --model-uri ${MODEL_URI}
```

`--picollm-access-key ${PICOLLM_ACCESS_KEY}`

## Results

### Perplexity

The table below depicts the perplexity of the original models.

<table>
<tbody>
  <tr>
    <td>Model</td>
    <td>Perplexity</td>
  </tr>
  <tr>
    <td>Llama-3-8b 16.1G</td>
    <td>11.61</td>
  </tr>
  <tr>
    <td>Llama-2-7b 13.5G</td>
    <td>8.40</td>
  </tr>
  <tr>
    <td>Mistral-7b-v0.1 15.0G</td>
    <td>10.50</td>
  </tr>
  <tr>
    <td>Gemma-7b 17.1G</td>
    <td>14.67</td>
  </tr>
  <tr>
    <td>Phi-2 5.6G</td>
    <td>17.38</td>
  </tr>
  <tr>
    <td>Gemma-2b 5.0G</td>
    <td>16.79</td>
  </tr>
</tbody>
</table>

The table below depicts the perplexity of the quantized models.

<table>
<tbody>
  <tr>
    <td>Model</td>
    <td>GPTQ</td>
    <td>picoLLM</td>
  </tr>
  <tr>
    <td>Llama-3-8b 5.7G</td>
    <td>12.31</td>
    <td><strong>11.73</strong></td>
  </tr>
  <tr>
    <td>Llama-3-8b 4.9G</td>
    <td>17.47</td>
    <td><strong>11.90</strong></td>
  </tr>
  <tr>
    <td>Llama-3-8b 4.0G</td>
    <td>712.70</td>
    <td><strong>12.67</strong></td>
  </tr>
  <tr>
    <td>Llama-2-7b 3.9G</td>
    <td>8.59</td>
    <td><strong>8.50</strong></td>
  </tr>
  <tr>
    <td>Llama-2-7b 3.1G</td>
    <td>9.66</td>
    <td><strong>8.86</strong></td>
  </tr>
  <tr>
    <td>Llama-2-7b 2.3G</td>
    <td>67.43</td>
    <td><strong>10.87</strong></td>
  </tr>
  <tr>
    <td>Mistral-7b-v0.1 4.2G</td>
    <td><strong>10.43</strong></td>
    <td>10.62</td>
  </tr>
  <tr>
    <td>Mistral-7b-v0.1 3.3G</td>
    <td>2909.83</td>
    <td><strong>10.81</strong></td>
  </tr>
  <tr>
    <td>Mistral-7b-v0.1 2.4G</td>
    <td>1176.43</td>
    <td><strong>14.87</strong></td>
  </tr>
  <tr>
    <td>Gemma-7b 7.2G</td>
    <td>15.47</td>
    <td><strong>14.82</strong></td>
  </tr>
  <tr>
    <td>Gemma-7b 6.2G</td>
    <td>27.29</td>
    <td><strong>14.84</strong></td>
  </tr>
  <tr>
    <td>Gemma-7b 5.2G</td>
    <td>33370970.40</td>
    <td><strong>15.08</strong></td>
  </tr>
  <tr>
    <td>Phi-2 1.8G</td>
    <td>18.15</td>
    <td><strong>17.76</strong></td>
  </tr>
  <tr>
    <td>Phi-2 1.5G</td>
    <td>19.94</td>
    <td><strong>18.14</strong></td>
  </tr>
  <tr>
    <td>Phi-2 1.2G</td>
    <td>76.55</td>
    <td><strong>20.22</strong></td>
  </tr>
  <tr>
    <td>Gemma-2b 3.1G</td>
    <td>17.85</td>
    <td><strong>16.86</strong></td>
  </tr>
  <tr>
    <td>Gemma-2b 2.9G</td>
    <td>24.11</td>
    <td><strong>16.86</strong></td>
  </tr>
  <tr>
    <td>Gemma-2b 2.6G</td>
    <td>8377.74</td>
    <td><strong>16.86</strong></td>
  </tr>
</tbody>
</table>

### ARC Easy

The table below depicts the ARC (easy) score of the original models.

<table>
<tbody>
  <tr>
    <td>Model</td>
    <td>ARC-E</td>
  </tr>
  <tr>
    <td>Llama-3-8b 16.1G</td>
    <td>75.80</td>
  </tr>
  <tr>
    <td>Llama-2-7b 13.5G</td>
    <td>44.87</td>
  </tr>
  <tr>
    <td>Mistral-7b-v0.1 15.0G</td>
    <td>80.56</td>
  </tr>
  <tr>
    <td>Gemma-7b 17.1G</td>
    <td>75.51</td>
  </tr>
  <tr>
    <td>Phi-2 5.6G</td>
    <td>75.25</td>
  </tr>
  <tr>
    <td>Gemma-2b 5.0G</td>
    <td>33.75</td>
  </tr>
</tbody>
</table>

The table below depicts the ARC (easy) score of the quantized models.

<table>
<tbody>
  <tr>
    <td>Model</td>
    <td>GPTQ</td>
    <td>picoLLM</td>
  </tr>
  <tr>
    <td>Llama-3-8b 5.7G</td>
    <td>72.85</td>
    <td><strong>78.83</strong></td>
  </tr>
  <tr>
    <td>Llama-3-8b 4.9G</td>
    <td>43.39</td>
    <td><strong>77.02</strong></td>
  </tr>
  <tr>
    <td>Llama-3-8b 4.0G</td>
    <td>24.71</td>
    <td><strong>71.76</strong></td>
  </tr>
  <tr>
    <td>Llama-2-7b 3.9G</td>
    <td>39.23</td>
    <td><strong>41.96</strong></td>
  </tr>
  <tr>
    <td>Llama-2-7b 3.1G</td>
    <td>32.95</td>
    <td><strong>33.96</strong></td>
  </tr>
  <tr>
    <td>Llama-2-7b 2.3G</td>
    <td>23.91</td>
    <td><strong>24.49</strong></td>
  </tr>
  <tr>
    <td>Mistral-7b-v0.1 4.2G</td>
    <td><strong>77.27</strong></td>
    <td>73.95</td>
  </tr>
  <tr>
    <td>Mistral-7b-v0.1 3.3G</td>
    <td>23.91</td>
    <td><strong>72.10</strong></td>
  </tr>
  <tr>
    <td>Mistral-7b-v0.1 2.4G</td>
    <td>24.92</td>
    <td><strong>46.46</strong></td>
  </tr>
  <tr>
    <td>Gemma-7b 7.2G</td>
    <td>76.52</td>
    <td><strong>84.18</strong></td>
  </tr>
  <tr>
    <td>Gemma-7b 6.2G</td>
    <td>44.28</td>
    <td><strong>84.51</strong></td>
  </tr>
  <tr>
    <td>Gemma-7b 5.2G</td>
    <td>23.95</td>
    <td><strong>84.13</strong></td>
  </tr>
  <tr>
    <td>Phi-2 1.8G</td>
    <td>70.45</td>
    <td><strong>75.04</strong></td>
  </tr>
  <tr>
    <td>Phi-2 1.5G</td>
    <td>56.61</td>
    <td><strong>70.66</strong></td>
  </tr>
  <tr>
    <td>Phi-2 1.2G</td>
    <td>22.10</td>
    <td><strong>62.42</strong></td>
  </tr>
  <tr>
    <td>Gemma-2b 3.1G</td>
    <td>30.39</td>
    <td><strong>34.39</strong></td>
  </tr>
  <tr>
    <td>Gemma-2b 2.9G</td>
    <td>24.37</td>
    <td><strong>34.39</strong></td>
  </tr>
  <tr>
    <td>Gemma-2b 2.6G</td>
    <td>23.82</td>
    <td><strong>34.39</strong></td>
  </tr>
</tbody>
</table>

### ARC Challenge

The table below depicts the ARC (challenge) score of the original models.

<table>
<tbody>
  <tr>
    <td>Model</td>
    <td>ARC-C</td>
  </tr>
  <tr>
    <td>Llama-3-8b 16.1G</td>
    <td>63.05</td>
  </tr>
  <tr>
    <td>Llama-2-7b 13.5G</td>
    <td>37.03</td>
  </tr>
  <tr>
    <td>Mistral-7b-v0.1 15.0G</td>
    <td>67.49</td>
  </tr>
  <tr>
    <td>Gemma-7b 17.1G</td>
    <td>64.93</td>
  </tr>
  <tr>
    <td>Phi-2 5.6G</td>
    <td>61.60</td>
  </tr>
  <tr>
    <td>Gemma-2b 5.0G</td>
    <td>30.38</td>
  </tr>
</tbody>
</table>

The table below depicts the ARC (challenge) score of the quantized models.

<table>
<tbody>
  <tr>
    <td>Model</td>
    <td>GPTQ</td>
    <td>picoLLM</td>
  </tr>
  <tr>
    <td>Llama-3-8b 5.7G</td>
    <td>60.24</td>
    <td><strong>64.33</strong></td>
  </tr>
  <tr>
    <td>Llama-3-8b 4.9G</td>
    <td>36.18</td>
    <td><strong>63.48</strong></td>
  </tr>
  <tr>
    <td>Llama-3-8b 4.0G</td>
    <td>23.29</td>
    <td><strong>57.85</strong></td>
  </tr>
  <tr>
    <td>Llama-2-7b 3.9G</td>
    <td>32.42</td>
    <td><strong>34.30</strong></td>
  </tr>
  <tr>
    <td>Llama-2-7b 3.1G</td>
    <td>27.56</td>
    <td><strong>28.24</strong></td>
  </tr>
  <tr>
    <td>Llama-2-7b 2.3G</td>
    <td>21.16</td>
    <td><strong>23.63</strong></td>
  </tr>
  <tr>
    <td>Mistral-7b-v0.1 4.2G</td>
    <td>64.42</td>
    <td>60.49</td>
  </tr>
  <tr>
    <td>Mistral-7b-v0.1 3.3G</td>
    <td>24.06</td>
    <td><strong>59.04</strong></td>
  </tr>
  <tr>
    <td>Mistral-7b-v0.1 2.4G</td>
    <td>23.21</td>
    <td><strong>37.80</strong></td>
  </tr>
  <tr>
    <td>Gemma-7b 7.2G</td>
    <td>66.30</td>
    <td><strong>72.35</strong></td>
  </tr>
  <tr>
    <td>Gemma-7b 6.2G</td>
    <td>33.62</td>
    <td><strong>72.35</strong></td>
  </tr>
  <tr>
    <td>Gemma-7b 5.2G</td>
    <td>24.06</td>
    <td><strong>72.61</strong></td>
  </tr>
  <tr>
    <td>Phi-2 1.8G</td>
    <td>57.42</td>
    <td><strong>62.46</strong></td>
  </tr>
  <tr>
    <td>Phi-2 1.5G</td>
    <td>44.97</td>
    <td><strong>57.51</strong></td>
  </tr>
  <tr>
    <td>Phi-2 1.2G</td>
    <td>24.49</td>
    <td><strong>47.87</strong></td>
  </tr>
  <tr>
    <td>Gemma-2b 3.1G</td>
    <td>26.37</td>
    <td><strong>30.97</strong></td>
  </tr>
  <tr>
    <td>Gemma-2b 2.9G</td>
    <td>23.55</td>
    <td><strong>30.97</strong></td>
  </tr>
  <tr>
    <td>Gemma-2b 2.6G</td>
    <td>24.83</td>
    <td><strong>30.97</strong></td>
  </tr>
</tbody>
</table>
