SAVED_MODEL=/tmp/cls/checkpoint-500
INDEX_MAP=index_to_name.json
CUSTOM_HANDLER=./custom_handler.py
MODEL_STORE=./model_store

mkdir -p $MODEL_STORE
torch-model-archiver -f   \
--model-name "bert-base"  \
--version 1.0  \
--serialized-file $SAVED_MODEL/pytorch_model.bin  \
--extra-files "$SAVED_MODEL/config.json,$SAVED_MODEL/vocab.txt,${INDEX_MAP}"  \
--export-path=$MODEL_STORE \
--handler=$CUSTOM_HANDLER
