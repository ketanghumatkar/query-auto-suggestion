# Installation
`pip install -r requirements.txt`

# Run
`python service.py`



# FAQ

<h2 align="center">:speech_balloon: FAQ</h2>
<p align="right"><a href="#bert-as-service"><sup>▴ Back to top</sup></a></p>

[![ReadTheDoc](https://readthedocs.org/projects/bert-as-service/badge/?version=latest&style=for-the-badge)](https://bert-as-service.readthedocs.io/en/latest/section/faq.html)

##### **Q:** Do you have a paper or other written explanation to introduce your model's details?

The design philosophy and technical details can be found [in my blog post](https://hanxiao.github.io/2019/01/02/Serving-Google-BERT-in-Production-using-Tensorflow-and-ZeroMQ/).

##### **Q:** Where is the BERT code come from?

**A:** [BERT code of this repo](server/bert_serving/server/bert/) is forked from the [original BERT repo](https://github.com/google-research/bert) with necessary modification, [especially in extract_features.py](server/bert_serving/server/bert/extract_features.py).

##### **Q:** How large is a sentence vector?
In general, each sentence is translated to a 768-dimensional vector. Depending on the pretrained BERT you are using, `pooling_strategy` and `pooling_layer` the dimensions of the output vector could be different.

##### **Q:** How do you get the fixed representation? Did you do pooling or something?

**A:** Yes, pooling is required to get a fixed representation of a sentence. In the default strategy `REDUCE_MEAN`, I take the second-to-last hidden layer of all of the tokens in the sentence and do average pooling.

##### **Q:** Are you suggesting using BERT without fine-tuning?

**A:** Yes and no. On the one hand, Google pretrained BERT on Wikipedia data, thus should encode enough prior knowledge of the language into the model. Having such feature is not a bad idea. On the other hand, these prior knowledge is not specific to any particular domain. It should be totally reasonable if the performance is not ideal if you are using it on, for example, classifying legal cases. Nonetheless, you can always first fine-tune your own BERT on the downstream task and then use `bert-as-service` to extract the feature vectors efficiently. Keep in mind that `bert-as-service` is just a feature extraction service based on BERT. Nothing stops you from using a fine-tuned BERT.

##### **Q:** Can I get a concatenation of several layers instead of a single layer ?

**A:** Sure! Just use a list of the layer you want to concatenate when calling the server. Example:

```bash
bert-serving-start -pooling_layer -4 -3 -2 -1 -model_dir /tmp/english_L-12_H-768_A-12/
```

##### **Q:** What are the available pooling strategies?

**A:** Here is a table summarizes all pooling strategies I implemented. Choose your favorite one by specifying `bert-serving-start -pooling_strategy`.

|Strategy|Description|
|---|---|
| `NONE` | no pooling at all, useful when you want to use word embedding instead of sentence embedding. This will results in a `[max_seq_len, 768]` encode matrix for a sequence.|
| `REDUCE_MEAN` | take the average of the hidden state of encoding layer on the time axis |
| `REDUCE_MAX` | take the maximum of the hidden state of encoding layer on the time axis |
| `REDUCE_MEAN_MAX` | do `REDUCE_MEAN` and `REDUCE_MAX` separately and then concat them together on the last axis, resulting in 1536-dim sentence encodes |
| `CLS_TOKEN` or `FIRST_TOKEN` | get the hidden state corresponding to `[CLS]`, i.e. the first token |
| `SEP_TOKEN` or `LAST_TOKEN` | get the hidden state corresponding to `[SEP]`, i.e. the last token |

##### **Q:** Why not use the hidden state of the first token as default strategy, i.e. the `[CLS]`?

**A:** Because a pre-trained model is not fine-tuned on any downstream tasks yet. In this case, the hidden state of `[CLS]` is not a good sentence representation. If later you fine-tune the model, you may use `[CLS]` as well.

##### **Q:** BERT has 12/24 layers, so which layer are you talking about?

**A:** By default this service works on the second last layer, i.e. `pooling_layer=-2`. You can change it by setting `pooling_layer` to other negative values, e.g. -1 corresponds to the last layer.

##### **Q:** Why not the last hidden layer? Why second-to-last?

**A:** The last layer is too closed to the target functions (i.e. masked language model and next sentence prediction) during pre-training, therefore may be biased to those targets. If you question about this argument and want to use the last hidden layer anyway, please feel free to set `pooling_layer=-1`.

##### **Q:** So which layer and which pooling strategy is the best?

**A:** It depends. Keep in mind that different BERT layers capture different information. To see that more clearly, here is a visualization on [UCI-News Aggregator Dataset](https://www.kaggle.com/uciml/news-aggregator-dataset), where I randomly sample 20K news titles; get sentence encodes from different layers and with different pooling strategies, finally reduce it to 2D via PCA (one can of course do t-SNE as well, but that's not my point). There are only four classes of the data, illustrated in red, blue, yellow and green. To reproduce the result, please run [example7.py](example/example7.py).

<p align="center"><img src=".github/pool_mean.png?raw=true"></p>

<p align="center"><img src=".github/pool_max.png?raw=true"></p>

Intuitively, `pooling_layer=-1` is close to the training output, so it may be biased to the training targets. If you don't fine tune the model, then this could lead to a bad representation. `pooling_layer=-12` is close to the word embedding, may preserve the very original word information (with no fancy self-attention etc.). On the other hand, you may achieve the very same performance by simply using a word-embedding only. That said, anything in-between [-1, -12] is then a trade-off.

##### **Q:** Could I use other pooling techniques?

**A:** For sure. But if you introduce new `tf.variables` to the graph, then you need to train those variables before using the model. You may also want to check [some pooling techniques I mentioned in my blog post](https://hanxiao.github.io/2018/06/24/4-Encoding-Blocks-You-Need-to-Know-Besides-LSTM-RNN-in-Tensorflow/#pooling-block).

##### **Q:** Do I need to batch the data before `encode()`?

No, not at all. Just do `encode` and let the server handles the rest. If the batch is too large, the server will do batching automatically and it is more efficient than doing it by yourself. No matter how many sentences you have, 10K or 100K, as long as you can hold it in client's memory, just send it to the server. Please also read [the benchmark on the client batch size](https://github.com/hanxiao/bert-as-service#speed-wrt-client_batch_size).


##### **Q:** Can I start multiple clients and send requests to one server simultaneously?

**A:** Yes! That's the purpose of this repo. In fact you can start as many clients as you want. One server can handle all of them (given enough time).

##### **Q:** How many requests can one service handle concurrently?

**A:** The maximum number of concurrent requests is determined by `num_worker` in `bert-serving-start`. If you a sending more than `num_worker` requests concurrently, the new requests will be temporally stored in a queue until a free worker becomes available.

##### **Q:** So one request means one sentence?

**A:** No. One request means a list of sentences sent from a client. Think the size of a request as the batch size. A request may contain 256, 512 or 1024 sentences. The optimal size of a request is often determined empirically. One large request can certainly improve the GPU utilization, yet it also increases the overhead of transmission. You may run `python example/example1.py` for a simple benchmark.

##### **Q:** How about the speed? Is it fast enough for production?

**A:** It highly depends on the `max_seq_len` and the size of a request. On a single Tesla M40 24GB with `max_seq_len=40`, you should get about 470 samples per second using a 12-layer BERT. In general, I'd suggest smaller `max_seq_len` (25) and larger request size (512/1024).

##### **Q:** Did you benchmark the efficiency?

**A:** Yes. See [Benchmark](#zap-benchmark).

To reproduce the results, please run `bert-serving-benchmark`.

##### **Q:** What is backend based on?

**A:** [ZeroMQ](http://zeromq.org/).

##### **Q:** What is the parallel processing model behind the scene?

<img src=".github/bert-parallel-pipeline.png?raw=true" width="600">

##### **Q:** Why does the server need two ports?
One port is for pushing text data into the server, the other port is for publishing the encoded result to the client(s). In this way, we get rid of back-chatter, meaning that at every level recipients never talk back to senders. The overall message flow is strictly one-way, as depicted in the above figure. Killing back-chatter is essential to real scalability, allowing us to use `BertClient` in an asynchronous way.

##### **Q:** Do I need Tensorflow on the client side?

**A:** No. Think of `BertClient` as a general feature extractor, whose output can be fed to *any* ML models, e.g. `scikit-learn`, `pytorch`, `tensorflow`. The only file that client need is [`client.py`](service/client.py). Copy this file to your project and import it, then you are ready to go.

##### **Q:** Can I use multilingual BERT model provided by Google?

**A:** Yes.

##### **Q:** Can I use my own fine-tuned BERT model?

**A:** Yes. In fact, this is suggested. Make sure you have the following three items in `model_dir`:

- A TensorFlow checkpoint (`bert_model.ckpt`) containing the pre-trained weights (which is actually 3 files).
- A vocab file (`vocab.txt`) to map WordPiece to word id.
- A config file (`bert_config.json`) which specifies the hyperparameters of the model.

##### **Q:** Can I run it in python 2?

**A:** Server side no, client side yes. This is based on the consideration that python 2.x might still be a major piece in some tech stack. Migrating the whole downstream stack to python 3 for supporting `bert-as-service` can take quite some effort. On the other hand, setting up `BertServer` is just a one-time thing, which can be even [run in a docker container](#run-bert-service-on-nvidia-docker). To ease the integration, we support python 2 on the client side so that you can directly use `BertClient` as a part of your python 2 project, whereas the server side should always be hosted with python 3.

##### **Q:** Do I need to do segmentation for Chinese?

No, if you are using [the pretrained Chinese BERT released by Google](https://github.com/google-research/bert#pre-trained-models) you don't need word segmentation. As this Chinese BERT is character-based model. It won't recognize word/phrase even if you intentionally add space in-between. To see that more clearly, this is what the BERT model actually receives after tokenization:

```python
bc.encode(['hey you', 'whats up?', '你好么？', '我 还 可以'])
```

```
tokens: [CLS] hey you [SEP]
input_ids: 101 13153 8357 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
input_mask: 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

tokens: [CLS] what ##s up ? [SEP]
input_ids: 101 9100 8118 8644 136 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
input_mask: 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

tokens: [CLS] 你 好 么 ？ [SEP]
input_ids: 101 872 1962 720 8043 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
input_mask: 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

tokens: [CLS] 我 还 可 以 [SEP]
input_ids: 101 2769 6820 1377 809 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
input_mask: 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
```

That means the word embedding is actually the character embedding for Chinese-BERT.


##### **Q:** Why my (English) word is tokenized to `##something`?

Because your word is out-of-vocabulary (OOV). The tokenizer from Google uses a greedy longest-match-first algorithm to perform tokenization using the given vocabulary.

For example:
```python
input = "unaffable"
tokenizer_output = ["un", "##aff", "##able"]
```

##### **Q:** Can I use my own tokenizer?

Yes. If you already tokenize the sentence on your own, simply send use `encode` with `List[List[Str]]` as input and turn on `is_tokenized`, i.e. `bc.encode(texts, is_tokenized=True)`.


##### **Q:** I encounter `zmq.error.ZMQError: Operation cannot be accomplished in current state` when using `BertClient`, what should I do?

**A:** This is often due to the misuse of `BertClient` in multi-thread/process environment. Note that you can’t reuse one `BertClient` among multiple threads/processes, you have to make a separate instance for each thread/process. For example, the following won't work at all:

```python
# BAD example
bc = BertClient()

# in Proc1/Thread1 scope:
bc.encode(lst_str)

# in Proc2/Thread2 scope:
bc.encode(lst_str)
```

Instead, please do:

```python
# in Proc1/Thread1 scope:
bc1 = BertClient()
bc1.encode(lst_str)

# in Proc2/Thread2 scope:
bc2 = BertClient()
bc2.encode(lst_str)
```

##### **Q:** After running the server, I have several garbage `tmpXXXX` folders. How can I change this behavior ?

**A:** These folders are used by ZeroMQ to store sockets. You can choose a different location by setting the environment variable `ZEROMQ_SOCK_TMP_DIR` :
`export ZEROMQ_SOCK_TMP_DIR=/tmp/`

##### **Q:** The cosine similarity of two sentence vectors is unreasonably high (e.g. always > 0.8), what's wrong?

**A:** A decent representation for a downstream task doesn't mean that it will be meaningful in terms of cosine distance. Since cosine distance is a linear space where all dimensions are weighted equally. if you want to use cosine distance anyway, then please focus on the rank not the absolute value. Namely, do not use:
```
if cosine(A, B) > 0.9, then A and B are similar
```
Please consider the following instead:
```
if cosine(A, B) > cosine(A, C), then A is more similar to B than C.
```

The graph below illustrates the pairwise similarity of 3000 Chinese sentences randomly sampled from web (char. length < 25). We compute cosine similarity based on the sentence vectors and [Rouge-L](https://en.wikipedia.org/wiki/ROUGE_(metric)) based on the raw text. The diagonal (self-correlation) is removed for the sake of clarity. As one can see, there is some positive correlation between these two metrics.

<p align="center"><img src=".github/cosine-vs-rougel.png?raw=true"/></p>


##### **Q:** I'm getting bad performance, what should I do?

**A:** This often suggests that the pretrained BERT could not generate a decent representation of your downstream task. Thus, you can fine-tune the model on the downstream task and then use `bert-as-service` to serve the fine-tuned BERT. Note that, `bert-as-service` is just a feature extraction service based on BERT. Nothing stops you from using a fine-tuned BERT.

##### **Q:** Can I run the server side on CPU-only machine?

**A:** Yes, please run `bert-serving-start -cpu -max_batch_size 16`. Note that, CPUs do not scale as well as GPUs to large batches, therefore the `max_batch_size` on the server side needs to be smaller, e.g. 16 or 32.

##### **Q:** How can I choose `num_worker`?

**A:** Generally, the number of workers should be less than or equal to the number of GPUs or CPUs that you have. Otherwise, multiple workers will be allocated to one GPU/CPU, which may not scale well (and may cause out-of-memory on GPU).

##### **Q:** Can I specify which GPU to use?

**A:** Yes, you can specifying `-device_map` as follows:
```bash
bert-serving-start -device_map 0 1 4 -num_worker 4 -model_dir ...
```
This will start four workers and allocate them to GPU0, GPU1, GPU4 and again GPU0, respectively. In general, if `num_worker` > `device_map`, then devices will be reused and shared by the workers (may scale suboptimally or cause OOM); if `num_worker` < `device_map`, only `device_map[:num_worker]` will be used.

Note, `device_map` is ignored when running on CPU.

<h2 align="center">:zap: Benchmark</h2>
<p align="right"><a href="#bert-as-service"><sup>▴ Back to top</sup></a></p>

[![ReadTheDoc](https://readthedocs.org/projects/bert-as-service/badge/?version=latest&style=for-the-badge)](https://bert-as-service.readthedocs.io/en/latest/section/benchmark.html)

The primary goal of benchmarking is to test the scalability and the speed of this service, which is crucial for using it in a dev/prod environment. Benchmark was done on Tesla M40 24GB, experiments were repeated 10 times and the average value is reported.

To reproduce the results, please run
```bash
bert-serving-benchmark --help
```

Common arguments across all experiments are:

| Parameter         | Value |
|-------------------|-------|
| num_worker        | 1,2,4 |
| max_seq_len       | 40    |
| client_batch_size | 2048  |
| max_batch_size    | 256   |
| num_client        | 1     |

#### Speed wrt. `max_seq_len`

`max_seq_len` is a parameter on the server side, which controls the maximum length of a sequence that a BERT model can handle. Sequences larger than `max_seq_len` will be truncated on the left side. Thus, if your client want to send long sequences to the model, please make sure the server can handle them correctly.

Performance-wise, longer sequences means slower speed and  more chance of OOM, as the multi-head self-attention (the core unit of BERT) needs to do dot products and matrix multiplications between every two symbols in the sequence.

<img src=".github/max_seq_len.png?raw=true" width="600">

| `max_seq_len` | 1 GPU | 2 GPU | 4 GPU |
|---------------|-------|-------|-------|
| 20            | 903   | 1774  | 3254  |
| 40            | 473   | 919   | 1687  |
| 80            | 231   | 435   | 768   |
| 160           | 119   | 237   | 464   |
| 320           | 54    | 108   | 212   |

#### Speed wrt. `client_batch_size`

`client_batch_size` is the number of sequences from a client when invoking `encode()`. For performance reason, please consider encoding sequences in batch rather than encoding them one by one.

For example, do:
```python
# prepare your sent in advance
bc = BertClient()
my_sentences = [s for s in my_corpus.iter()]
# doing encoding in one-shot
vec = bc.encode(my_sentences)
```

DON'T:
```python
bc = BertClient()
vec = []
for s in my_corpus.iter():
    vec.append(bc.encode(s))
```

It's even worse if you put `BertClient()` inside the loop. Don't do that.

<img src=".github/client_batch_size.png?raw=true" width="600">

| `client_batch_size` | 1 GPU | 2 GPU | 4 GPU |
|---------------------|-------|-------|-------|
| 1                   | 75    | 74    | 72    |
| 4                   | 206   | 205   | 201   |
| 8                   | 274   | 270   | 267   |
| 16                  | 332   | 329   | 330   |
| 64                  | 365   | 365   | 365   |
| 256                 | 382   | 383   | 383   |
| 512                 | 432   | 766   | 762   |
| 1024                | 459   | 862   | 1517  |
| 2048                | 473   | 917   | 1681  |
| 4096                | 481   | 943   | 1809  |



#### Speed wrt. `num_client`
`num_client` represents the number of concurrent clients connected to the server at the same time.

<img src=".github/num_clients.png?raw=true" width="600">

| `num_client` | 1 GPU | 2 GPU | 4 GPU |
|--------------|-------|-------|-------|
| 1            | 473   | 919   | 1759  |
| 2            | 261   | 512   | 1028  |
| 4            | 133   | 267   | 533   |
| 8            | 67    | 136   | 270   |
| 16           | 34    | 68    | 136   |
| 32           | 17    | 34    | 68    |

As one can observe, 1 clients 1 GPU = 381 seqs/s, 2 clients 2 GPU 402 seqs/s, 4 clients 4 GPU 413 seqs/s. This shows the efficiency of our parallel pipeline and job scheduling, as the service can leverage the GPU time  more exhaustively as concurrent requests increase.


#### Speed wrt. `max_batch_size`

`max_batch_size` is a parameter on the server side, which controls the maximum number of samples per batch per worker. If a incoming batch from client is larger than `max_batch_size`, the server will split it into small batches so that each of them is less or equal than `max_batch_size` before sending it to workers.

<img src=".github/max_batch_size.png?raw=true" width="600">

| `max_batch_size` | 1 GPU | 2 GPU | 4 GPU |
|------------------|-------|-------|-------|
| 32               | 450   | 887   | 1726  |
| 64               | 459   | 897   | 1759  |
| 128              | 473   | 931   | 1816  |
| 256              | 473   | 919   | 1688  |
| 512              | 464   | 866   | 1483  |


#### Speed wrt. `pooling_layer`

`pooling_layer` determines the encoding layer that pooling operates on. For example, in a 12-layer BERT model, `-1` represents the layer closed to the output, `-12` represents the layer closed to the embedding layer. As one can observe below, the depth of the pooling layer affects the speed.

<img src=".github/pooling_layer.png?raw=true" width="600">

| `pooling_layer` | 1 GPU | 2 GPU | 4 GPU |
|-----------------|-------|-------|-------|
| [-1]            | 438   | 844   | 1568  |
| [-2]            | 475   | 916   | 1686  |
| [-3]            | 516   | 995   | 1823  |
| [-4]            | 569   | 1076  | 1986  |
| [-5]            | 633   | 1193  | 2184  |
| [-6]            | 711   | 1340  | 2430  |
| [-7]            | 820   | 1528  | 2729  |
| [-8]            | 945   | 1772  | 3104  |
| [-9]            | 1128  | 2047  | 3622  |
| [-10]           | 1392  | 2542  | 4241  |
| [-11]           | 1523  | 2737  | 4752  |
| [-12]           | 1568  | 2985  | 5303  |


#### Speed wrt. `-fp16` and `-xla`

`bert-as-service` supports two additional optimizations: half-precision and XLA, which can be turned on by adding `-fp16` and `-xla` to `bert-serving-start`, respectively. To enable these two options, you have to meet the following requirements:

- your GPU supports FP16 instructions;
- your Tensorflow is self-compiled with XLA and `-march=native`;
- your CUDA and cudnn are not too old.

On Tesla V100 with `tensorflow=1.13.0-rc0` it gives:

<img src=".github/fp16-xla.svg" width="600">

FP16 achieves ~1.4x speedup (round-trip) comparing to the FP32 counterpart. To reproduce the result, please run `python example/example1.py`.


<h2 align="center">Citing</h2>
<p align="right"><a href="#bert-as-service"><sup>▴ Back to top</sup></a></p>

If you use bert-as-service in a scientific publication, we would appreciate references to the following BibTex entry:

```latex
@misc{xiao2018bertservice,
  title={bert-as-service},
  author={Xiao, Han},
  howpublished={\url{https://github.com/hanxiao/bert-as-service}},
  year={2018}
}
```
