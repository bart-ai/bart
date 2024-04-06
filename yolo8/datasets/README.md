# bart-ai: datasets

In order to download the datasets used for this project you must first use the associated scripts

```sh
ROBOFLOW_API_KEY="XXX" python3 ./download-datasets.py
python3 ./download-datasets.py
```

This will populate the datasets directory, which can then be used alltogether with the root `data.yaml` file, or we can selectively decide which datasets to use by manually editing the file (and not calling the nested `data.yaml` within each directory, as those won't work due to the path name)