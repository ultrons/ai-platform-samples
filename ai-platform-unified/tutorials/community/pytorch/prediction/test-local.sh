get_prediction () {
    cat > instances.json <<END
{
  "instances": [
    {
      "data": {
        "b64": "$(base64 --wrap=0 sample-review.txt)"
      }
    }
  ]
}
END

    curl -X POST \
      -H "Content-Type: application/json; charset=utf-8" \
      -d @instances.json \
      localhost:7080/predictions/bert-base
}

# Start torchserve locally
torchserve  \
--start  \
--model-store ./model_store  \
--models bert-base=bert-base.mar  \
--ts-config  config.properties

# Delay to allow model to be loaded in torchserve
sleep 60

# Send a prediction request
get_prediction

# Stop the local server
torchserve  \
--stop

# Test container
docker run -d -p 7080:7080  \
--name=bert-base us-central1-docker.pkg.dev/pytorch-tpu-nfs/pytorch-models/bert-base:latest;

# Delay to allow model to be loaded in torchserve
sleep 60
get_prediction

docker stop bert-base
docker rm bert-base
