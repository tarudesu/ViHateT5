MODEL_ID=$1
RESULT_FILEPATH=$2

python ./t5_evaluate.py \
  --model_id=$MODEL_ID \
  --result_filepath=$RESULT_FILEPATH

# bash t5_evaluate.sh <MODEL_ID of HF model> <RESULT_FILEPATH>
```

