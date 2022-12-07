# Neural Semantic Search via Transformer Embeddings

## Requirements
- pytorch
- numpy
- scipy
- scikit-learn
- pandas
- matplotlib
- plotly
- openai
- transformers (huggingface)
- elasticsearch

Creating the features index in elastic search:

```
PUT /features
{
  "mappings": {
    "properties": {
      "embeddings": {
        "type": "dense_vector",
        "dims": 768
      }
    }
  }
}
```
