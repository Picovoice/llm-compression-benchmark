# LLM Compression Benchmark

Made in Vancouver, Canada by [Picovoice](https://picovoice.ai)

This repository is a minimalist and extensible framework for benchmarking LLM compression algorithms.

## Table of Contents

- [Algorithms](#algorithms)
    - [GPTQ](#gptq)
    - [picoLLM Compression](#picollm-compression)
- [Tasks](#tasks)
    - [MMLU Score](#mmlu-score)
    - [ARC Score](#arc-score)
    - [Perplexity Loss](#perplexity-loss)
- [Data](#data)
    - [MMLU](#mmlu)
    - [ARC](#arc)
    - [Perplexity (C4)](#perplexity-c4)
    - [Quantization (C4)](#quantization-c4)
- [Models](#models)
- [Usage](#usage)
- [Results](#results)
    - [MMLU](#mmlu-1)
    - [ARC-Easy](#arc-easy)
    - [ARC-Challenge](#arc-challenge)
    - [Perplexity](#perplexity)

## Algorithms

### GPTQ

[GPTQ](https://arxiv.org/abs/2210.17323) is arguably the most popular quantization algorithm for LLMs. GPTQ fully
reconstructs weights so that the quantized version closely mimics the full-precision one.

### picoLLM Compression

picoLLM Compression is Picovoice's in-house LLM compression algorithm. Given a target size, picoLLM optimally
distributes available bits within and across LLM's weights.

## Tasks

### MMLU Score

[MMLU](https://huggingface.co/datasets/lukaemon/mmlu) (Massive Multitask Language Understanding) is a
multiple-choice dataset that can measure the models' ability to understand natural language.

### ARC Score

[ARC]((https://allenai.org/data/arc)) (AI2 Reasoning Challenge) is a multiple-choice dataset that measures
the models' reasoning ability. The ARC dataset has two partitions: `Easy` and `Challenge`. We perform the benchmark on
both partitions and report the results separately.

### Perplexity Loss

Perplexity measures the models' language modeling capabilities. Research has shown that perplexity is very sensitive to
quantization and can be used to detect deterioration in the model's output distribution early on.

## Data

All required data for the benchmark is available under the `/res` folder But if you wish to reproduce it find out
how the data is curated or change it you can follow the sections below.

### MMLU

Download the [MMLU dataset](https://huggingface.co/datasets/lukaemon/mmlu) and run the following from the
repository's root to extract and format it:

```console
python3 data/mmlu.py --dataset-folder ${DATASET_FOLDER}
```

### ARC

Download the [ARC dataset](https://allenai.org/data/arc) and run the following from the repository's root to extract and
format the `Challenge` portion:

```console
python3 data/arc.py --dataset-folder ${DATASET_FOLDER}
```

Perform the above for the `Easy` portion:

```console
python3 data/arc.py --dataset-folder ${DATASET_FOLDER} --easy
```

### Perplexity (C4)

For the perplexity measurement, we use 128 randomly selected text snippets from the validation portion of the
[C4 dataset](https://huggingface.co/datasets/c4). Once you download the dataset, run the following from the root of the
repository to extract and normalize the data:

```console
python3 data/c4-normalize.py \
--repository-folder ${REPOSITORY_FOLDER} \
--normalized-folder ${VALIDATION_FOLDER} \
--portion validation
```

Replace `${REPOSITORY_FOLDER}` with the path to the downloaded dataset repository and `${VALIDATION_FOLDER}` with a
folder to hold onto the normalized data.

Then we sample 128 sequences from the normalized data:

```console
python3 data/c4-sample.py \
--dataset-folder ${VALIDATION_FOLDER} \
--portion valid
```

### Quantization (C4)

We need a sample dataset for quantization algorithms (GPTQ, picoLLM). We use 128 randomly selected text snippets from
the train portion of the [C4 dataset](https://huggingface.co/datasets/c4). Once you download the dataset, run the
following from the root of the repository to extract and normalize the data:

```console
python3 data/c4-normalize.py \
--repository-folder ${REPOSITORY_FOLDER} \
--normalized-folder ${TRAIN_FOLDER} \
--portion train
```

Replace `${REPOSITORY_FOLDER}` with the path to the downloaded dataset repository and `${TRAIN_FOLDER}` with a
folder to hold onto the normalized data.

Then we sample 128 sequences from the normalized data:

```console
python3 data/c4-sample.py \
--dataset-folder ${TRAIN_FOLDER} \
--portion train
```

## Models

We use six models:

- `Gemma-2b`
- `Gemma-7b`
- `Llama-2-7b`
- `Llama-3-8b`
- `Mistral-7b-v0.1`
- `Phi-2`

The corresponding picoLLM compressed models are on [Picovoice Console](https://console.picovoice.ai/). We create GPTQ
models using the package [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ). You can quantize the models by running the
following:

```console
python3 model/autogptq.py \
--model-uri ${MODEL_URI} \
--quantized-model-folder ${QUANTIZED_MODEL_FOLDER} \
--bits ${BITS}
```

## Usage

To measure the MMLU score for a given model, run the following:

```console
python3 mmlu.py \
--compression ${COMPRESSION} \
--model-uri ${MODEL_URI}
```

Replace `${COMPRESSION}` with the model's compression. i.e., `NONE` for full-precision models, `GPTQ,` or `picoLLM.`

To measure the ARC score for a given model, run the following:

```console
python3 arc.py \
--compression ${COMPRESSION} \
--model-uri ${MODEL_URI}
```

Replace `${COMPRESSION}` with the model's compression. i.e., `NONE` for full-precision models, `GPTQ,` or `picoLLM.`

To measure the perplexity for a given model, run the following:

```console
python3 perplexity.py \
--compression ${COMPRESSION} \
--model-uri ${MODEL_URI}
```

Replace `${COMPRESSION}` with the model's compression. i.e., `NONE` for full-precision models, `GPTQ,` or `picoLLM.`

When running picoLLM Compressed models, you must also provide your Picovoice AccessKey, which is available on
[Picovoice Console](https://console.picovoice.ai/).

```console
... --picollm-access-key ${PICOLLM_ACCESS_KEY}
```

## Results

Below are our benchmark results comparing GPTQ against picoLLM for all [models](model). We perform 2, 3, and 4-bit
quantization using GPTQ, then find the model size in GB and set that as the target size for picoLLM Compression. Hence,
both models have the same size in terms of the number of bytes. When performing GPTQ, we set the group size parameter to
128, set the damp percent to 0.1 and enabled activation reordering.

### MMLU

The table below depicts the MMLU score of the original models.

<table>
<tbody>
  <tr>
    <td>Model</td>
    <td>MMLU</td>
  </tr>
  <tr>
    <td>Gemma-2b 5.0G</td>
    <td>40.21</td>
  </tr>
  <tr>
    <td>Gemma-7b 17.1G</td>
    <td>64.48</td>
  </tr>
  <tr>
    <td>Llama-3-8b 16.1G</td>
    <td>64.88</td>
  </tr>
  <tr>
    <td>Llama-2-7b 13.5G</td>
    <td>46.38</td>
  </tr>
  <tr>
    <td>Mistral-7b-v0.1 15.0G</td>
    <td>62.41</td>
  </tr>
  <tr>
    <td>Phi-2 5.6G</td>
    <td>56.04</td>
  </tr>
</tbody>
</table>

The table below depicts the MMLU score of the quantized models.

<table>
<tbody>
  <tr>
    <td>Model</td>
    <td>GPTQ</td>
    <td>picoLLM</td>
  </tr>
<tr>
    <td>Gemma-2b 3.1G</td>
    <td>39.07</td>
    <td><strong>41.12</strong></td>
  </tr>
  <tr>
    <td>Gemma-2b 2.9G</td>
    <td>27.51</td>
    <td><strong>41.12</strong></td>
  </tr>
  <tr>
    <td>Gemma-2b 2.6G</td>
    <td>24.93</td>
    <td><strong>41.12</strong></td>
  </tr>
  <tr>
    <td>Gemma-7b 7.2G</td>
    <td>62.58</td>
    <td><strong>64.98</strong></td>
  </tr>
  <tr>
    <td>Gemma-7b 6.2G</td>
    <td>53.30</td>
    <td><strong>64.57</strong></td>
  </tr>
  <tr>
    <td>Gemma-7b 5.2G</td>
    <td>25.58</td>
    <td><strong>64.32</strong></td>
  </tr>
  <tr>
    <td>Llama-2-7b 3.9G</td>
    <td><strong>45.26</strong></td>
    <td>44.99</td>
  </tr>
  <tr>
    <td>Llama-2-7b 3.1G</td>
    <td>40.40</td>
    <td><strong>40.68</strong></td>
  </tr>
  <tr>
    <td>Llama-2-7b 2.3G</td>
    <td>25.36</td>
    <td><strong>28.72</strong></td>
  </tr>
  <tr>
    <td>Llama-3-8b 5.7G</td>
    <td>63.09</td>
    <td><strong>64.96</strong></td>
  </tr>
  <tr>
    <td>Llama-3-8b 4.9G</td>
    <td>53.86</td>
    <td><strong>64.76</strong></td>
  </tr>
  <tr>
    <td>Llama-3-8b 4.0G</td>
    <td>25.05</td>
    <td><strong>61.26</strong></td>
  </tr>
  <tr>
    <td>Mistral-7b-v0.1 4.2G</td>
    <td><strong>61.00</strong></td>
    <td>59.19</td>
  </tr>
  <tr>
    <td>Mistral-7b-v0.1 3.3G</td>
    <td>23.73</td>
    <td><strong>57.72</strong></td>
  </tr>
  <tr>
    <td>Mistral-7b-v0.1 2.4G</td>
    <td>25.70</td>
    <td><strong>43.53</strong></td>
  </tr>
  <tr>
    <td>Phi-2 1.8G</td>
    <td><strong>54.61</strong></td>
    <td>54.11</td>
  </tr>
  <tr>
    <td>Phi-2 1.5G</td>
    <td>50.64</td>
    <td><strong>52.24</strong></td>
  </tr>
  <tr>
    <td>Phi-2 1.2G</td>
    <td>26.05</td>
    <td><strong>48.86</strong></td>
  </tr>
</tbody>
</table>

### ARC Easy

The table below depicts the ARC Easy score of the original models.

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

The table below depicts the ARC Easy score of the quantized models.

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

The table below depicts the ARC Challenge score of the original models.

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

The table below depicts the ARC Challenge score of the quantized models.

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
    <td><strong>64.42</strong></td>
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
