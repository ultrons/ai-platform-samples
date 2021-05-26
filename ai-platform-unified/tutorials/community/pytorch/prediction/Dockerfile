FROM pytorch/torchserve:0.3.0-cpu

COPY model_store /home/model-server/model_store
COPY config.properties /home/model-server/

RUN pip install transformers==4.3.3

USER root
USER model-server

CMD ["torchserve", \
     "--start", \
     "--model-store=/home/model-server/model_store", \
     "--ts-config=/home/model-server/config.properties",      \
     "--models",      \
     "bert-base=bert-base.mar" \
    ]
