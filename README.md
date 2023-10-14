# tprompt-llm

```
pip install ctranslate2
```

```
pip install --upgrade huggingface_hub
```

```
pip install accelerate
```


## Manually do this: https://github.com/OpenNMT/CTranslate2/issues/1501

```
ct2-transformers-converter --model mistralai/Mistral-7B-v0.1 --quantization int8 --output_dir ./models/ctranslate2 --low_cpu_mem_usage
```
