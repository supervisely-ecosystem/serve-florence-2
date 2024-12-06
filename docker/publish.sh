cp ../dev_requirements.txt . && \
cp ../src/models.json . && \
docker build -t supervisely/florence-2:1.0.0 . && \
docker push supervisely/florence-2:1.0.0